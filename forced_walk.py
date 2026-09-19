# -*- coding: utf-8 -*-

import os

# Set these BEFORE importing tensorflow
# threads = "3" # Adjust this to leave 1-2 cores free
# os.environ["OMP_NUM_THREADS"] = threads
# os.environ["OPENBLAS_NUM_THREADS"] = threads
# os.environ["MKL_NUM_THREADS"] = threads
# os.environ["VECLIB_MAXIMUM_THREADS"] = threads
# os.environ["NUMEXPR_NUM_THREADS"] = threads

import tensorflow as tf
# tf.config.threading.set_intra_op_parallelism_threads(int(threads))
# tf.config.threading.set_inter_op_parallelism_threads(int(threads))

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
import gc
import random
import contextlib
import logging
import yaml  
from typing import Callable, List, Tuple, Optional, Any, Dict, Union

# Leave 1 or 2 threads for the OS. 
# tf.config.threading.set_intra_op_parallelism_threads(6)
# tf.config.threading.set_inter_op_parallelism_threads(6)

# Disable TF info logs to speed up console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(2)

print(f"TensorFlow Version: {tf.__version__}")

logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)

# Avoid adding multiple handlers if the module is reloaded
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class ForcedWalkTrial:
    """
    Represents a single evaluation trial, passing suggested parameters 
    to the black-box objective function.
    """
    def __init__(self, values_dict: Optional[Dict[str, Any]] = None):
        self.is_discovery = values_dict is None
        self._values = values_dict or {}
        self.parameters_config: List[Tuple[str, Union[Tuple[float, float], List[Any]], str]] = []

    def suggest_int(self, name: str, low: int, high: int) -> int:
        if low > high:
            raise ValueError(f"In '{name}', lower bound ({low}) cannot be > upper bound ({high}).")
            
        if self.is_discovery:
            self.parameters_config.append((name, (low, high), "int"))
            val = random.randint(low, high)
            self._values[name] = val
            return val
        return int(self._values[name])

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        if low > high:
            raise ValueError(f"In '{name}', lower bound ({low}) cannot be > upper bound ({high}).")
            
        if self.is_discovery:
            self.parameters_config.append((name, (low, high), "float"))
            val = round(random.uniform(low, high), 5)
            self._values[name] = val
            return val
        return float(self._values[name])

    def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
        if not choices:
            raise ValueError(f"Categorical parameter '{name}' must have at least one choice.")
            
        if self.is_discovery:
            self.parameters_config.append((name, choices, "categorical"))
            val = random.choice(choices)
            self._values[name] = val
            return val
        return self._values[name]


class ForcedWalkStudy:
    """Orchestrates the interface and execution of the Forced Walk optimization algorithm."""
    
    def __init__(self, direction: str = "minimize", terminate_value: Optional[float] = None, hyperparams: Optional[Dict[str, Any]] = None, config_path: str = "config.yaml"):
        if direction not in ["minimize", "maximize"]:
            raise ValueError("Direction must be either 'minimize' or 'maximize'.")
            
        self.direction = direction
        self.terminate_value = terminate_value
        self.best_value: Optional[float] = None
        
        self.best_score = float('inf') if direction == "minimize" else float('-inf')
        self.scaler: Optional[MinMaxScaler] = None
        self.training_data: List[List[Any]] = []     
        self.global_model: Optional[tf.keras.Model] = None    
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
            
        with open(config_path, "r") as file:
            self.training_params = yaml.safe_load(file)
            
        if self.training_params is None:
            self.training_params = {}
        
        if hyperparams is not None:
            allowed_keys = set(self.training_params.keys())
            provided_keys = set(hyperparams.keys())
            rogue_keys = provided_keys - allowed_keys
            
            if rogue_keys:
                raise ValueError(f"Unrecognized hyperparameters: {rogue_keys}. Valid keys: {allowed_keys}")
                
            self.training_params.update(hyperparams)
            
        logging_config = self.training_params.get("logging", "True")
        self.logging_enabled = str(logging_config).strip().lower() == "true"

        colors_config = self.training_params.get("use_colors", "True")
        self.use_colors = str(colors_config).strip().lower() == "true"
            
        self._validate_training_params()

    def _validate_training_params(self) -> None:
        p = self.training_params
        if not (0 < p["search_radius"] <= 0.5):
            raise ValueError(f"'search_radius' must be (0, 0.5]. Got: {p['search_radius']}")
        if not isinstance(p["beta"], int) or p["beta"] <= 0:
            raise ValueError(f"'beta' must be a positive integer. Got: {p['beta']}")
        if not isinstance(p["tau"], int) or p["tau"] <= 0:
            raise ValueError(f"'tau' must be a positive integer. Got: {p['tau']}")
        if p["zeta"] <= 1 and p["zeta"] != 0: 
            raise ValueError(f"'zeta' must be > 1 (or 0 to disable). Got: {p['zeta']}")
        if not (0 <= p["mu"] < 1):
            raise ValueError(f"'mu' must be in [0, 1). Got: {p['mu']}")
        if not isinstance(p["rho"], int) or p["rho"] < 3:
            raise ValueError(f"'rho' must be strictly > 2. Got: {p['rho']}")
        if not (0 < p["lambda"] <= 1):
            raise ValueError(f"'lambda' must be in (0, 1]. Got: {p['lambda']}")

    @contextlib.contextmanager
    def _device_context(self):
        if self.training_params["force_cpu"]:
            with tf.device('/CPU:0'):
                yield
        else:
            yield

    @staticmethod
    def _remove_duplicates(input_list: List[List[Any]]) -> List[List[Any]]:
        if not input_list:
            return []
            
        seen = set()
        unique_list = []
        for item in input_list:
            t_item = tuple(item)
            if t_item not in seen:
                seen.add(t_item)
                unique_list.append(item)
        return unique_list

    def _init_global_model(self, xdim: int, ydim: int) -> None:
        if xdim <= 0 or ydim <= 0:
            raise ValueError(f"Model dims must be > 0. Got xdim={xdim}, ydim={ydim}")
        
        nodes = self.training_params["nn_nodes"]
        activation = self.training_params["nn_activation"]
        lr = self.training_params["nn_learning_rate"]
        
        num_layers = self.training_params.get("nn_layers", 1)

        with self._device_context():
            model_layers = [tf.keras.Input(shape=(xdim,))]
            
            for _ in range(num_layers):
                model_layers.append(layers.Dense(nodes, activation=activation))
                
            model_layers.append(layers.Dense(ydim))
            
            self.global_model = tf.keras.Sequential(model_layers)
            opt = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=False)
            self.global_model.compile(loss='mean_squared_error', optimizer=opt)

    def _reset_weights(self, model: tf.keras.Model) -> None:
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                init_config = layer.kernel_initializer.get_config()
                init_config['seed'] = random.randint(0, 10**6)
                new_init = layer.kernel_initializer.__class__.from_config(init_config)
                layer.kernel.assign(new_init(layer.kernel.shape))
                
            if hasattr(layer, 'bias_initializer') and layer.bias is not None:
                layer.bias.assign(layer.bias_initializer(layer.bias.shape))

        if hasattr(model, 'optimizer') and model.optimizer is not None:
            for var in model.optimizer.variables():
                var.assign(tf.zeros_like(var))

    def _train_value_network(self) -> None:
        # 1. Apply sliding window factor (mu)
        not_use = self.training_params["mu"]
        delete_older = int(len(self.training_data) * not_use)
        data = self.training_data[delete_older:]
        
        # 2. Apply max_samples truncation constraint (if configured)
        max_samples_config = self.training_params.get("max_samples", "False")
        if str(max_samples_config).strip().lower() != "false":
            max_samples_limit = int(max_samples_config)
            if len(data) > max_samples_limit:
                # Keep only the latest `max_samples_limit` elements
                data = data[-max_samples_limit:]
        
        if len(data) == 0:
            return

        xdim = len(data[0]) - 1
        ydim = 1

        if self.global_model is None:
            self._init_global_model(xdim, ydim)
        
        self._reset_weights(self.global_model)

        with self._device_context():
            npa = np.asarray(data, dtype=np.float32)
            X = npa[:, 0:xdim]
            y = npa[:, xdim]

            min_val = np.min(y)
            if min_val <= 0:
                transformed_y = y + (np.abs(min_val) + 0.01)
            else:
                transformed_y = y

            self.scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = self.scaler.fit_transform(X).astype(np.float32)

            scaler2 = MinMaxScaler(feature_range=(0, 1))
            y_log = np.log(transformed_y).reshape(-1, 1)
            y_normalized = scaler2.fit_transform(y_log).astype(np.float32)

            epochs = self.training_params["nn_epochs_early"] if len(data) < 1000 else self.training_params["nn_epochs_late"]
                
            early_stop_config = self.training_params.get("early_stopping", "False")
            use_early_stopping = str(early_stop_config).strip().lower() == "true"
            
            callbacks_list = []
            if use_early_stopping:
                early_stop_cb = tf.keras.callbacks.EarlyStopping(
                    monitor='loss', 
                    patience=10, 
                    restore_best_weights=True
                )
                callbacks_list.append(early_stop_cb)

            self.global_model.fit(
                X_scaled, y_normalized,
                epochs=epochs,
                batch_size=self.training_params["nn_batch_size"],
                verbose=0,
                callbacks=callbacks_list
            )

    def _filter_moves(self, new_init: List[List[Any]], allow_params: int, parameters: List[Any]) -> List[List[Any]]:
        if allow_params <= 0 or not new_init:
            return []
        
        raw_data = np.array(new_init, dtype=np.float32)
        feature_columns = []
        
        for i, param_spec in enumerate(parameters):
            col_data = raw_data[:, i]
            if param_spec[2] == "categorical":
                num_categories = len(param_spec[1])
                row_indices = np.arange(len(col_data))
                one_hot = np.zeros((len(col_data), num_categories))
                one_hot[row_indices, col_data.astype(int)] = 1
                feature_columns.append(one_hot)
            else:
                feature_columns.append(col_data.reshape(-1, 1))
                
        X_new = np.hstack(feature_columns)
        X_new_scaled = self.scaler.transform(X_new)
        
        predictions = self.global_model(X_new_scaled, training=False).numpy().flatten()
        k = min(allow_params, len(predictions))
        
        if len(predictions) <= k:
            return new_init
        
        if self.direction == "minimize":
            best_indices_unsorted = np.argpartition(predictions, k-1)[:k]
            subset_preds = predictions[best_indices_unsorted]
            best_indices = best_indices_unsorted[np.argsort(subset_preds)]
        else:
            best_indices_unsorted = np.argpartition(predictions, -k)[-k:]
            subset_preds = predictions[best_indices_unsorted]
            best_indices = best_indices_unsorted[np.argsort(subset_preds)[::-1]]

        return [new_init[i] for i in best_indices]

    def _generate_candidates_vectorized(self, base_param: List[Any], num_candidates: int, dice_min: int, dice_max: int, scale: float, parameters: List[Any]) -> List[List[Any]]:
        if dice_min >= dice_max + 1:
            raise ValueError(f"dice_min ({dice_min}) must be <= dice_max ({dice_max})")
            
        num_params = len(parameters)
        low_limits, high_limits, steps, is_categorical, is_int = [], [], [], [], []
        
        base_categorical_scale = self.training_params["base_scale"]
        for i, p in enumerate(parameters):
            p_type = p[2]
            if p_type == "categorical":
                low, high = 0, len(p[1])
                s = (high - low) / base_categorical_scale
                is_categorical.append(True)
                is_int.append(True)
            else:
                low, high = p[1][0], p[1][1]
                s = (high - low) / scale
                is_categorical.append(False)
                is_int.append(p_type == "int")
                
            low_limits.append(low)
            high_limits.append(high)
            steps.append(s)
            
        low_limits = np.array(low_limits)
        high_limits = np.array(high_limits)
        steps = np.array(steps)
        
        base_arr = np.array(base_param)
        dice_rolls = np.random.randint(dice_min, dice_max + 1, size=(num_candidates, num_params))
        multipliers = np.random.choice([1, -1], size=(num_candidates, num_params))
        
        shifts = dice_rolls * multipliers * steps
        new_params = base_arr + shifts
        
        new_params = np.where(new_params < low_limits, high_limits + (new_params - low_limits), new_params)
        new_params = np.where(new_params > high_limits, low_limits + (new_params - high_limits), new_params)
        
        for col in range(num_params):
            if is_int[col]:
                new_params[:, col] = np.trunc(new_params[:, col])
                if is_categorical[col]:
                    max_cat_index = len(parameters[col][1]) - 1
                    new_params[:, col] = np.clip(new_params[:, col], 0, max_cat_index)
            else: 
                new_params[:, col] = np.round(new_params[:, col], 8)
                
        final_params = new_params.tolist()
        for row in final_params:
            for col in range(num_params):
                if is_int[col]:
                    row[col] = int(row[col])
                    
        return final_params

    def _global_sampling_r(self, filtration_total: int, dim: int, run_number: int) -> int:
        midpoint, steepness, max_value = 20, 0.2, 40
        sigmoid = max_value / (1 + np.exp(-steepness * (run_number - midpoint)))
        return int(sigmoid * dim) + filtration_total

    def _init_parameters(self, parameters: List[Any]) -> List[Any]:
        init_param = []
        for i in range(len(parameters)):
            p_type = parameters[i][2]
            limits = parameters[i][1]
            
            if p_type == "float":
                init = round(random.uniform(limits[0], limits[1]), 5)
            elif p_type == "int":
                init = random.randint(limits[0], limits[1])
            elif p_type == "categorical":
                init = random.randint(0, len(limits) - 1)
            init_param.append(init)
        return init_param

    def _generate_parameters(self, init: List[Any], run_number: int, scale: float, parameters: List[Any]) -> List[List[Any]]:
        low_high_dice = 1
        high_high_dice = self.training_params["search_radius"] * self.training_params["base_scale"]
        filtration_total = self.training_params["beta"]
        
        phase2_low_dice = 1
        phase2_high_dice = max(1, int(high_high_dice * self.training_params["lambda"]))
        phase2_batch_size = self.training_params["R_local"] 
        
        dim = len(init)

        random_gen_count = self._global_sampling_r(filtration_total, dim, run_number)
        candidates_phase_1 = self._generate_candidates_vectorized(
            init, random_gen_count, low_high_dice, high_high_dice, scale, parameters
        )
        best_phase_1 = self._filter_moves(candidates_phase_1, filtration_total, parameters)
        
        if not best_phase_1: 
            return [init]

        all_final_candidates = []
        for pivot_point in best_phase_1:
            candidates_phase_2 = self._generate_candidates_vectorized(
                pivot_point, phase2_batch_size, phase2_low_dice, phase2_high_dice, scale, parameters
            )
            branch_best = self._filter_moves(candidates_phase_2, 1, parameters)
            all_final_candidates.extend(branch_best)

        return self._remove_duplicates(all_final_candidates)

    def _append_training_data(self, row: List[Any], value: float) -> None:
        if (self.direction == "minimize" and value < self.best_score) or \
           (self.direction == "maximize" and value > self.best_score):
            self.best_score = value
        self.training_data.append(row + [value])

    def _forced_walk(self, max_iterations: int, parameters: List[Any], get_values: Callable[[List[Any]], float], initial_run_data: Optional[Tuple[List[Any], float]] = None) -> float:
        scale = self.training_params["base_scale"] 
        base_scale_ref = self.training_params["base_scale"]
        
        threshold_metric = self.training_params["tau"]
        scale_factor = self.training_params["zeta"]
        random_start_count = self.training_params["rho"]
        
        self.training_data = [] 
        self.best_score = float('inf') if self.direction == "minimize" else float('-inf')
        best_init, current_run, best_metric_counter = [], 1, 0

        def encode_parameters(raw_params: List[Any]) -> Tuple[List[Any], List[float]]:
            transformed, training_row = [], []
            for i, spec in enumerate(parameters):
                val = raw_params[i]
                if spec[2] == "categorical":
                    idx = spec[1].index(val) if val in spec[1] else max(0, min(int(val), len(spec[1])-1))
                    transformed.append(spec[1][idx])
                    one_hot = [1 if k == idx else 0 for k in range(len(spec[1]))]
                    training_row.extend(one_hot)
                else:
                    transformed.append(val)
                    training_row.append(val)
            return transformed, training_row

        def evaluate_and_update(raw_params: List[Any], current_best_init: List[Any], precomputed_val: Optional[float] = None) -> Tuple[float, List[Any], bool]:
            c_red = "\033[1;31m" if self.use_colors else ""
            c_blue = "\033[1;34m" if self.use_colors else ""
            c_green = "\033[1;32m" if self.use_colors else ""
            c_yellow = "\033[1;33m" if self.use_colors else ""
            c_reset = "\033[0m" if self.use_colors else ""

            transformed, training_row = encode_parameters(raw_params)                    
            val = precomputed_val if precomputed_val is not None else get_values(transformed)
            
            is_improvement = (self.direction == "minimize" and val < self.best_score) or \
                             (self.direction == "maximize" and val > self.best_score)
                             
            if is_improvement:
                current_best_init = raw_params
                
            if self.logging_enabled:
                if is_improvement:
                    logger.info(f">>> Run {c_red}{current_run}{c_reset}| Params: {c_blue}{transformed}{c_reset} | New Best Value: {c_green}{val}{c_reset}")
                else:
                    logger.info(f">>> Run {c_red}{current_run}{c_reset}| Params: {c_blue}{transformed}{c_reset} | Value: {c_yellow}{val}{c_reset}")
                
            self._append_training_data(training_row, val)
            
            should_stop = False
            if self.terminate_value is not None:
                if (self.direction == "minimize" and val <= self.terminate_value) or \
                   (self.direction == "maximize" and val >= self.terminate_value):
                    should_stop = True
                    
            return val, current_best_init, should_stop

        if initial_run_data:
            raw_p, val = initial_run_data
            val, best_init, stop = evaluate_and_update(raw_p, best_init, precomputed_val=val)
            current_run += 1
            if stop: return self.best_score

        warmup_remaining = random_start_count - (1 if initial_run_data else 0)
        for _ in range(max(0, warmup_remaining)):
            val, best_init, stop = evaluate_and_update(self._init_parameters(parameters), best_init)
            current_run += 1
            if stop: return self.best_score

        self._train_value_network()

        while current_run <= (max_iterations + (1 if initial_run_data else 0)):
            new_candidates = self._generate_parameters(best_init, current_run - random_start_count, scale, parameters)
            
            for candidate in new_candidates:
                if current_run > (max_iterations + (1 if initial_run_data else 0)):
                    break
                    
                val, best_init, stop = evaluate_and_update(candidate, best_init)
                
                if val == self.best_score: 
                    best_metric_counter = 0  
                else: 
                    best_metric_counter += 1  
                
                if scale_factor > 0 and best_metric_counter >= threshold_metric:
                    best_metric_counter = 0
                    interim_scale = int(scale * scale_factor)

                    if interim_scale < self.training_params["max_zoom"] * base_scale_ref:
                        scale = interim_scale
                        if self.logging_enabled:
                            print(f"Search Radius Constricted by a factor of {scale / base_scale_ref}")
                    else:
                        if self.logging_enabled:
                            print(f"Search Radius cannot exceed the max zoom limit of {self.training_params['max_zoom']}")
                                                                                                                        
                current_run += 1
                if stop: return self.best_score

            self._train_value_network()
            
        tf.keras.backend.clear_session()
        self.global_model = None 
        gc.collect()
        
        if self.logging_enabled:
            print(f"\n--- Final Validation ---\nAbsolute Best Parameters: {best_init}\nAbsolute Best Value: {self.best_score}")
            
        return self.best_score

    def optimize(self, objective_func: Callable[[ForcedWalkTrial], float], n_trials: int) -> None:
        if n_trials <= 0:
            raise ValueError(f"n_trials must be at least 1, received: {n_trials}")
            
        discovery_trial = ForcedWalkTrial()
        first_score = objective_func(discovery_trial)
        parameters = discovery_trial.parameters_config
        
        if not parameters:
            raise ValueError("No parameters detected. Did your objective function call any suggest_* methods on the trial?")
            
        param_names = [p[0] for p in parameters]
        first_params_raw = []
        for (name, choices, p_type) in parameters:
            val = discovery_trial._values[name]
            if p_type == "categorical":
                first_params_raw.append(choices.index(val))
            else:
                first_params_raw.append(val)
        
        def fw_objective(param_array: List[Any]) -> float:
            values_dict = {name: val for name, val in zip(param_names, param_array)}
            eval_trial = ForcedWalkTrial(values_dict)
            return objective_func(eval_trial)
            
        remaining_trials = max(1, n_trials - 1)
        self.best_value = self._forced_walk(
            max_iterations=remaining_trials,
            parameters=parameters,
            get_values=fw_objective,
            initial_run_data=(first_params_raw, first_score)
        )

        if self.direction == "minimize":
            self.best_value = min(self.best_value, first_score)
        else:
            self.best_value = max(self.best_value, first_score)


def create_fw_study(direction: str = "minimize", terminate_value: Optional[float] = None, hyperparams: Optional[Dict[str, Any]] = None, config_path: str = "config.yaml") -> ForcedWalkStudy:
    return ForcedWalkStudy(direction=direction, terminate_value=terminate_value, hyperparams=hyperparams, config_path=config_path)

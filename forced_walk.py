# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
import gc
import random
import os
import contextlib
import logging
from typing import Callable, List, Tuple, Optional, Any, Dict, Union

# Disable TF info logs to speed up console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"TensorFlow Version: {tf.__version__}")

logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)

# Avoid adding multiple handlers if the module is reloaded
if not logger.handlers:
    ch = logging.StreamHandler()
    # \033[1;31m = Bold Red | \033[0m = Reset
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class ForcedWalkTrial:
    """
    Represents a single evaluation trial, passing suggested parameters 
    to the black-box objective function.
    """
    def __init__(self, values_dict: Optional[Dict[str, Any]] = None):
        # If no dictionary is provided, this is the Discovery Phase (Trial 0)
        self.is_discovery = values_dict is None
        self._values = values_dict or {}
        self.parameters_config: List[Tuple[str, Union[Tuple[float, float], List[Any]], str]] = []

    def suggest_int(self, name: str, low: int, high: int) -> int:
        """Suggests an integer value for the parameter within the specified bounds."""
        if low > high:
            raise ValueError(f"In '{name}', lower bound ({low}) cannot be > upper bound ({high}).")
            
        if self.is_discovery:
            self.parameters_config.append((name, (low, high), "int"))
            val = random.randint(low, high)
            self._values[name] = val
            return val
        return int(self._values[name])

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        """Suggests a float value for the parameter within the specified bounds."""
        if low > high:
            raise ValueError(f"In '{name}', lower bound ({low}) cannot be > upper bound ({high}).")
            
        if self.is_discovery:
            self.parameters_config.append((name, (low, high), "float"))
            val = round(random.uniform(low, high), 5)
            self._values[name] = val
            return val
        return float(self._values[name])

    def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
        """Suggests a categorical value from the provided list of choices."""
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
    
    def __init__(self, direction: str = "minimize", terminate_value: Optional[float] = None, hyperparams: Optional[Dict[str, Any]] = None):
        if direction not in ["minimize", "maximize"]:
            raise ValueError("Direction must be either 'minimize' or 'maximize'.")
            
        self.direction = direction
        self.terminate_value = terminate_value
        self.best_value: Optional[float] = None
        self.use_colors = True
        
        # --- OOP State Encapsulation ---
        self.best_score = float('inf') if direction == "minimize" else float('-inf')
        self.scaler: Optional[MinMaxScaler] = None
        self.training_data: List[List[Any]] = []     # Experience Replay Buffer
        self.global_model: Optional[tf.keras.Model] = None    # Value Network Surrogate Policy
        
        # --- Algorithm & Neural Network Configurations ---
        self.training_params = {
            "search_radius": 0.5,    # Global exploration radius (max 0.5)
            "beta": 1,               # Evaluation batch size (selection bottleneck)
            "tau": 20,               # Stagnation limit for step-scaling
            "zeta": 2,               # Zoom factor for constricting trust region
            "mu": 0.0,               # Sliding window factor for pruning
            "max_zoom": 64,          # Maximum allowed zoom multiplier
            "rho": 3,                # Initial random samples for warm-up
            "R_local": 10000,        # Local sampling intensity
            "lambda": 0.02,          # Locality factor (proximal radius)
            "base_scale": 10000,     # Search grid resolution
            
            # --- Extracted Surrogate NN Hyperparameters ---
            "nn_activation": "relu",
            "nn_nodes": 32,
            "nn_learning_rate": 0.0008,
            "nn_batch_size": 4,
            "nn_epochs_early": 200,  # Epochs when data < 1000
            "nn_epochs_late": 300,   # Epochs when data >= 1000
            "force_cpu": True        # Forces TF to use CPU (avoids GPU memory overhead for small NNs)
        }
        
        if hyperparams is not None:
            allowed_keys = set(self.training_params.keys())
            provided_keys = set(hyperparams.keys())
            rogue_keys = provided_keys - allowed_keys
            
            if rogue_keys:
                raise ValueError(f"Unrecognized hyperparameters: {rogue_keys}. Valid keys: {allowed_keys}")
                
            self.training_params.update(hyperparams)
            
        self._validate_training_params()

    def _validate_training_params(self) -> None:
        """Validates mathematical boundaries of the configuration dictionary."""
        p = self.training_params
        if not (0 < p["search_radius"] <= 0.5):
            raise ValueError(f"'search_radius' must be (0, 0.5]. Got: {p['search_radius']}")
        if not isinstance(p["beta"], int) or p["beta"] <= 0:
            raise ValueError(f"'beta' must be a positive integer. Got: {p['beta']}")
        if not isinstance(p["tau"], int) or p["tau"] <= 0:
            raise ValueError(f"'tau' must be a positive integer. Got: {p['tau']}")
        if p["zeta"] <= 1 and p["zeta"] != 0: # Allowing 0 based on your ablation study table
            raise ValueError(f"'zeta' must be > 1 (or 0 to disable). Got: {p['zeta']}")
        if not (0 <= p["mu"] < 1):
            raise ValueError(f"'mu' must be in [0, 1). Got: {p['mu']}")
        if not isinstance(p["rho"], int) or p["rho"] < 3:
            raise ValueError(f"'rho' must be strictly > 2. Got: {p['rho']}")
        if not (0 < p["lambda"] <= 1):
            raise ValueError(f"'lambda' must be in (0, 1]. Got: {p['lambda']}")

    @contextlib.contextmanager
    def _device_context(self):
        """Context manager to optionally force CPU execution for the surrogate."""
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
        """Initializes the Neural Network surrogate architecture."""
        if xdim <= 0 or ydim <= 0:
            raise ValueError(f"Model dims must be > 0. Got xdim={xdim}, ydim={ydim}")
        
        nodes = self.training_params["nn_nodes"]
        activation = self.training_params["nn_activation"]
        lr = self.training_params["nn_learning_rate"]

        with self._device_context():
            self.global_model = tf.keras.Sequential([
                tf.keras.Input(shape=(xdim,)),
                layers.Dense(nodes, activation=activation),
                layers.Dense(nodes, activation=activation),
                layers.Dense(nodes, activation=activation),
                layers.Dense(ydim)
            ])
            opt = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=False)
            self.global_model.compile(loss='mean_squared_error', optimizer=opt)

    def _reset_weights(self, model: tf.keras.Model) -> None:
        """Re-initializes network weights to ensure ab initio training per epoch."""
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
        """Compiles the experience replay buffer and trains the NN surrogate off-policy."""
        not_use = self.training_params["mu"]
        delete_older = int(len(self.training_data) * not_use)
        data = self.training_data[delete_older:]
        
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
                
            self.global_model.fit(
                X_scaled, y_normalized,
                epochs=epochs,
                batch_size=self.training_params["nn_batch_size"],
                verbose=0  
            )

    def _filter_moves(self, new_init: List[List[Any]], allow_params: int, parameters: List[Any]) -> List[List[Any]]:
        """Filters the stochastically generated candidates through the NN surrogate."""
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
        """Vectorized stochastic perturbation to generate the global candidate pool."""
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
        
        # Enforce toroidal boundary conditions
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
        """Calculates the dynamically sized candidate pool via a Sigmoid curve."""
        midpoint, steepness, max_value = 20, 0.2, 40
        sigmoid = max_value / (1 + np.exp(-steepness * (run_number - midpoint)))
        return int(sigmoid * dim) + filtration_total

    def _init_parameters(self, parameters: List[Any]) -> List[Any]:
        """Generates a random initial coordinate within the search space."""
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
        """Executes the Bilevel Filtering Policy (Global Exploration + Proximal Refinement)."""
        low_high_dice = 1
        high_high_dice = self.training_params["search_radius"] * self.training_params["base_scale"]
        filtration_total = self.training_params["beta"]
        
        phase2_low_dice = 1
        phase2_high_dice = max(1, int(high_high_dice * self.training_params["lambda"]))
        phase2_batch_size = self.training_params["R_local"] 
        
        dim = len(init)

        # --- Stage 1: Global Exploration ---
        random_gen_count = self._global_sampling_r(filtration_total, dim, run_number)
        candidates_phase_1 = self._generate_candidates_vectorized(
            init, random_gen_count, low_high_dice, high_high_dice, scale, parameters
        )
        best_phase_1 = self._filter_moves(candidates_phase_1, filtration_total, parameters)
        
        if not best_phase_1: 
            return [init]

        # --- Stage 2: Proximal Refinement ---
        all_final_candidates = []
        for pivot_point in best_phase_1:
            candidates_phase_2 = self._generate_candidates_vectorized(
                pivot_point, phase2_batch_size, phase2_low_dice, phase2_high_dice, scale, parameters
            )
            branch_best = self._filter_moves(candidates_phase_2, 1, parameters)
            all_final_candidates.extend(branch_best)

        return self._remove_duplicates(all_final_candidates)

    def _append_training_data(self, row: List[Any], value: float) -> None:
        """Appends the state-value tuple to the Experience Replay Buffer."""
        if (self.direction == "minimize" and value < self.best_score) or \
           (self.direction == "maximize" and value > self.best_score):
            self.best_score = value
        self.training_data.append(row + [value])

    def _forced_walk(self, max_iterations: int, parameters: List[Any], get_values: Callable[[List[Any]], float], initial_run_data: Optional[Tuple[List[Any], float]] = None) -> float:
        """Main optimization loop managing evaluation, stagnation, and step-scaling."""
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

        # --- Integrated Discovery/Run 1 ---
        if initial_run_data:
            raw_p, val = initial_run_data
            val, best_init, stop = evaluate_and_update(raw_p, best_init, precomputed_val=val)
            current_run += 1
            if stop: return self.best_score

        # --- Stochastic Warm-Up Phase ---
        warmup_remaining = random_start_count - (1 if initial_run_data else 0)
        for _ in range(max(0, warmup_remaining)):
            val, best_init, stop = evaluate_and_update(self._init_parameters(parameters), best_init)
            current_run += 1
            if stop: return self.best_score

        # Initial Off-Policy Update
        self._train_value_network()

        # --- Main Optimization Loop ---
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
                
                # Adaptive Step-Scaling
                if scale_factor > 0 and best_metric_counter >= threshold_metric:
                    best_metric_counter = 0
                    interim_scale = int(scale * scale_factor)

                    if interim_scale < self.training_params["max_zoom"] * base_scale_ref:
                        scale = interim_scale
                        print(f"Search Radius Constricted by a factor of {scale / base_scale_ref}")
                    else:
                        print(f"Search Radius cannot exceed the max zoom limit of {self.training_params['max_zoom']}")
                                            
                current_run += 1
                if stop: return self.best_score

            self._train_value_network()
            
        # --- Teardown ---
        tf.keras.backend.clear_session()
        self.global_model = None 
        gc.collect()
        
        print(f"\n--- Final Validation ---\nAbsolute Best Parameters: {best_init}\nAbsolute Best Value: {self.best_score}")
        return self.best_score

    def optimize(self, objective_func: Callable[[ForcedWalkTrial], float], n_trials: int) -> None:
        """
        Executes the optimization process against the provided objective function.
        
        Args:
            objective_func: The black-box function to optimize.
            n_trials: Total number of function evaluations.
        """
        if n_trials <= 0:
            raise ValueError(f"n_trials must be at least 1, received: {n_trials}")
            
        # --- 1. Discovery Phase (Trial 0) ---
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
        
        # --- 2. Translation Phase ---
        def fw_objective(param_array: List[Any]) -> float:
            values_dict = {name: val for name, val in zip(param_names, param_array)}
            eval_trial = ForcedWalkTrial(values_dict)
            return objective_func(eval_trial)
            
        # --- 3. Execution Phase ---
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


# --- Global Helper Factory ---
def create_fw_study(direction: str = "minimize", terminate_value: Optional[float] = None, hyperparams: Optional[Dict[str, Any]] = None) -> ForcedWalkStudy:
    """Factory function to initialize and validate a new ForcedWalkStudy object."""
    return ForcedWalkStudy(direction=direction, terminate_value=terminate_value, hyperparams=hyperparams)

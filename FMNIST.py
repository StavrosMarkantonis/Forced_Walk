# -*- coding: utf-8 -*-
"""
Modified for Fashion-MNIST Dataset (Multi-class Classification)
Includes: Dropout, Weight Decay (L2), Global History Tracking, 
and High-Performance Custom Training Loops via AutoGraph.
"""
# !pip install ax-platform pyyaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces TensorFlow to ignore the GPU for faster MLP training
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import copy
import gc
import yaml

# Uncomment these as needed for your specific environment requirements:
import optuna
from ax.service.managed_loop import optimize
import forced_walk as fw

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

# --------------------------
# Global Variables (History & Data Tracking)
# --------------------------
ALL_METHODS_HISTORY = []
REPETITION = 0
TRIAL = 0
CURRENT_REP_SEED = None  

X_train_g, y_train_g, X_test_g, y_test_g = None, None, None, None

# --------------------------
# Configuration & YAML Loading
# --------------------------
CONFIG = {
    "RANDOM_SEED": None, 
    "TRIALS": 100,
    "REPETITIONS": 10,
    "TARGET_METRIC_NAME": "accuracy",  # Objective is now Accuracy
}
print("Fashion-MNIST Dynamic PARAMS (Maximization via High-Performance Loop)")

try:
    with open("config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)
        if yaml_config and "random_seed" in yaml_config:
            CONFIG["RANDOM_SEED"] = yaml_config["random_seed"]
            print(f"Loaded base random seed {CONFIG['RANDOM_SEED']} from config.yaml")
except FileNotFoundError:
    print("config.yaml not found. Proceeding without a base random seed.")

# --------------------------
# Global Hyperparameter Search Space
# --------------------------
# For the 5 hyperparameter experiments, batch size, learning rate, epochs, activation and dropout rate were used.
# Define the active search space here. 
# To disable tuning for a specific parameter, simply comment it out in HP_SPACE.
# The objective functions will automatically fall back to the static value defined in DEFAULT_HPARAMS.
HP_SPACE = {
    "batch_size": {"type": "int", "low": 16, "high": 1024},
    "learning_rate": {"type": "float", "low": 0.0005, "high": 0.002, "log": True},
    "epochs": {"type": "int", "low": 10, "high": 40},
    "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5},
    "activation": {"type": "categorical", "choices": ["relu", "sigmoid", "tanh"]},
    "optimizer_name": {"type": "categorical", "choices": ["Adam", "RMSprop", "SGD"]},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}
}

DEFAULT_HPARAMS = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 20,
    "dropout_rate": 0.0,
    "activation": "relu",
    "optimizer_name": "Adam",
    "weight_decay": 1e-4
}

# --- Dynamic Parser Helpers ---
def extract_optuna_params(trial):
    """Dynamically builds parameter dictionary from global HP_SPACE for Optuna/Forced Walk."""
    params = {}
    for hp_name, hp_conf in HP_SPACE.items():
        if hp_conf["type"] == "int":
            params[hp_name] = trial.suggest_int(hp_name, hp_conf["low"], hp_conf["high"])
        elif hp_conf["type"] == "float":
            params[hp_name] = trial.suggest_float(hp_name, hp_conf["low"], hp_conf["high"], log=hp_conf.get("log", False))
        elif hp_conf["type"] == "categorical":
            params[hp_name] = trial.suggest_categorical(hp_name, hp_conf["choices"])
    return params

def get_ax_search_space():
    """Dynamically translates global HP_SPACE into Ax parameter bounds list."""
    ax_params = []
    for hp_name, hp_conf in HP_SPACE.items():
        if hp_conf["type"] == "int":
            ax_params.append({"name": hp_name, "type": "range", "bounds": [hp_conf["low"], hp_conf["high"]], "value_type": "int"})
        elif hp_conf["type"] == "float":
            ax_params.append({"name": hp_name, "type": "range", "bounds": [hp_conf["low"], hp_conf["high"]], "value_type": "float", "log_scale": hp_conf.get("log", False)})
        elif hp_conf["type"] == "categorical":
            ax_params.append({"name": hp_name, "type": "choice", "values": hp_conf["choices"], "value_type": "str"})
    return ax_params

# --------------------------
# Load Base dataset
# --------------------------
print("Loading Fashion-MNIST Dataset into memory...")
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = keras.datasets.fashion_mnist.load_data()

# Combine them globally so we can split them cleanly per repetition using the shared seed
x_combined = np.concatenate((x_train_orig, x_test_orig))
y_combined = np.concatenate((y_train_orig, y_test_orig))

# Normalize and Flatten
X_raw = x_combined.astype(np.float32) / 255.0
X_raw = X_raw.reshape(-1, 28*28)
y_raw = y_combined.astype(np.int32)

# --------------------------
# Model builder
# --------------------------
def build_model(learning_rate, activation, optimizer_name, dropout_rate, weight_decay, rep_seed):
    """Constructs the Keras model and explicitly returns the model and optimizer for custom training."""
    l2_reg = regularizers.l2(weight_decay)
    kernel_init = keras.initializers.GlorotUniform(seed=rep_seed) if rep_seed is not None else "glorot_uniform"
    
    model = keras.Sequential([
        keras.layers.Input(shape=(28*28,)),
        
        keras.layers.Dense(128, activation=activation, kernel_regularizer=l2_reg, kernel_initializer=kernel_init),
        keras.layers.Dropout(dropout_rate, seed=rep_seed),
        
        # Output layer for 10 classes
        keras.layers.Dense(10, activation='softmax', kernel_regularizer=l2_reg, kernel_initializer=kernel_init)     
    ])
    
    if optimizer_name == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        
    return model, optimizer

def train_evaluate(batch_size, learning_rate, epochs, activation, optimizer_name, dropout_rate, weight_decay, X_tr, y_tr, X_te, y_te):
    """Executes a highly optimized custom training loop using AutoGraph and tf.data."""
    model, optimizer = build_model(
        learning_rate, activation, optimizer_name, dropout_rate, weight_decay, rep_seed=CURRENT_REP_SEED
    )
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    acc_metric = keras.metrics.SparseCategoricalAccuracy()
    
    # Pre-fetch data pipelines for speed
    train_dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(int(batch_size)).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_te, y_te)).batch(1024)

    # AutoGraph Compiled Training Step
    @tf.function
    def train_step(x, y, _model=model, _opt=optimizer):
        with tf.GradientTape() as tape:
            predictions = _model(x, training=True)
            loss = loss_fn(y, predictions)
            reg_loss = tf.math.add_n(_model.losses) if _model.losses else 0.0
            total_loss = loss + reg_loss
        gradients = tape.gradient(total_loss, _model.trainable_variables)
        _opt.apply_gradients(zip(gradients, _model.trainable_variables))
        return total_loss

    # AutoGraph Compiled Validation Step
    @tf.function
    def val_step(x, y, _model=model):
        predictions = _model(x, training=False)
        acc_metric.update_state(y, predictions)
        return loss_fn(y, predictions)

    for epoch in range(int(epochs)):
        # Training loop
        for x_batch, y_batch in train_dataset:
            train_loss = train_step(x_batch, y_batch)
            
            # Catch diverging gradients and prune the trial (return 0.0 accuracy)
            if tf.math.is_nan(train_loss):
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                return 0.0
        
    # Validation phase (run only at the final epoch to save time)
    acc_metric.reset_states()
    for x_batch_val, y_batch_val in val_dataset:
        val_step(x_batch_val, y_batch_val)
        
    final_accuracy = float(acc_metric.result().numpy())
        
    # Aggressive memory cleanup for the custom loop
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Return accuracy (Higher is better)
    return final_accuracy


# --------------------------
# Forced Walk Objective
# --------------------------
def get_Values_Forced_Walk(trial):
    global TRIAL
    
    current_hparams = copy.deepcopy(DEFAULT_HPARAMS)
    current_hparams.update(extract_optuna_params(trial))

    score = train_evaluate(
        **current_hparams,
        X_tr=X_train_g, y_tr=y_train_g, X_te=X_test_g, y_te=y_test_g
    )
    
    ALL_METHODS_HISTORY.append({
        "repetition": REPETITION,
        "trial": TRIAL,
        "method": "ForcedWalk",
        "score": score,
        "hparams": copy.deepcopy(current_hparams)
    })
    
    TRIAL += 1
    return score

def run_forced_walk(estimations):
    # MAXIMIZE FOR ACCURACY
    study = fw.create_fw_study(direction="maximize", terminate_value=None)
    study.optimize(get_Values_Forced_Walk, n_trials=estimations)
    return study.best_value


# --------------------------
# 1. Optuna Random Objective
# --------------------------
def optuna_random_objective(trial):
    global TRIAL 
    
    current_hparams = copy.deepcopy(DEFAULT_HPARAMS)
    current_hparams.update(extract_optuna_params(trial))
    
    score = train_evaluate(
        **current_hparams,
        X_tr=X_train_g, y_tr=y_train_g, X_te=X_test_g, y_te=y_test_g
    )
    
    ALL_METHODS_HISTORY.append({
        "repetition": REPETITION,
        "trial": TRIAL,
        "method": "OptunaRandom",
        "score": score,
        "hparams": copy.deepcopy(current_hparams)
    })
    
    TRIAL += 1
    return score

def run_optuna_random():
    sampler_seed = CURRENT_REP_SEED if CURRENT_REP_SEED is not None else None
    sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
    # MAXIMIZE FOR ACCURACY
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(optuna_random_objective, n_trials=CONFIG["TRIALS"])
    return study.best_value


# --------------------------
# 2. Optuna TPE Objective
# --------------------------
def optuna_tpe_objective(trial):
    global TRIAL
    
    current_hparams = copy.deepcopy(DEFAULT_HPARAMS)
    current_hparams.update(extract_optuna_params(trial))
    
    score = train_evaluate(
        **current_hparams,
        X_tr=X_train_g, y_tr=y_train_g, X_te=X_test_g, y_te=y_test_g
    )
    
    ALL_METHODS_HISTORY.append({
        "repetition": REPETITION,
        "trial": TRIAL,
        "method": "OptunaTPE",
        "score": score,
        "hparams": copy.deepcopy(current_hparams)
    })
    
    TRIAL += 1
    return score

def run_optuna_tpe():
    sampler_seed = CURRENT_REP_SEED if CURRENT_REP_SEED is not None else None
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    # MAXIMIZE FOR ACCURACY
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(optuna_tpe_objective, n_trials=CONFIG["TRIALS"])
    return study.best_value


# --------------------------
# 3. AX Optimization Objective
# --------------------------
def ax_objective(parameters):
    global TRIAL
    
    current_hparams = copy.deepcopy(DEFAULT_HPARAMS)
    current_hparams.update(parameters)
    
    score = train_evaluate(
        **current_hparams,
        X_tr=X_train_g, y_tr=y_train_g, X_te=X_test_g, y_te=y_test_g
    )
    
    ALL_METHODS_HISTORY.append({
        "repetition": REPETITION,
        "trial": TRIAL,
        "method": "Ax",
        "score": score,
        "hparams": copy.deepcopy(current_hparams)
    })
    
    TRIAL += 1
    return {CONFIG["TARGET_METRIC_NAME"]: (score, 0.0)}

def run_ax():
    ax_seed = CURRENT_REP_SEED if CURRENT_REP_SEED is not None else None
    best_parameters, values, experiment, model = optimize(
        parameters=get_ax_search_space(),
        evaluation_function=ax_objective,
        objective_name=CONFIG["TARGET_METRIC_NAME"],
        total_trials=CONFIG["TRIALS"],
        minimize=False,  # MAXIMIZE FOR ACCURACY
        random_seed=ax_seed
    )

    df = experiment.fetch_data().df
    # Sort descending so row 0 is the highest accuracy
    best_row = df.sort_values("mean", ascending=False).iloc[0]
    return best_row["mean"]


# --------------------------
# Print Stats Helper
# --------------------------
def calculate_stats(arr, name):
    if not arr:
        return
    np_arr = np.array(arr)
    clean_arr = np_arr[~np.isnan(np_arr)]
    mean = np.mean(clean_arr) if len(clean_arr) > 0 else float('nan')
    std_dev = np.std(clean_arr, ddof=1) if len(clean_arr) > 1 else 0.0
    print(f"{name} -> Runs: {clean_arr}")
    print(f"Mean: {mean:.6f}, Std Dev: {std_dev:.6f}\n")

def save_pbt_data(history, filename="history_results.pkl"):
    if not os.path.exists("ML"):
        os.makedirs("ML")
        
    path = "ML/" + filename
    with open(path, "wb") as f:
        pickle.dump(history, f)
    print(f"Data saved to {path}")


# --------------------------
# Main Execution Block
# --------------------------
if __name__ == "__main__":
    # SELECT METHODS TO RUN (Set to True to enable)
    RUN_CONFIG = {
        "Forced Walk":     {"enabled": True,  "runner": lambda: run_forced_walk(CONFIG["TRIALS"])},
        "Random Search":   {"enabled": False, "runner": run_optuna_random},
        "Optuna TPE":      {"enabled": False, "runner": run_optuna_tpe},
        "AX Optimization": {"enabled": False, "runner": run_ax}
    }
    
    results = {}

    for method_name, setup in RUN_CONFIG.items():
        if not setup["enabled"]:
            continue
            
        print(f"\n===== RUNNING {method_name.upper()} =====")
        
        print("Search Space Configuration:")
        if method_name == "AX Optimization":
            for param in get_ax_search_space():
                print(f"  {param}")
        else:
            for hp_name, hp_conf in HP_SPACE.items():
                print(f"  {hp_name}: {hp_conf}")
        print("--------------------------------------------------")

        results[method_name] = []
        REPETITION = 0 
        
        for t in range(CONFIG["REPETITIONS"]):
            TRIAL = 0 
            
            if CONFIG["RANDOM_SEED"] is not None:
                CURRENT_REP_SEED = CONFIG["RANDOM_SEED"] + t
            else:
                CURRENT_REP_SEED = None
                
            print(f"  -> Repetition {t + 1}/{CONFIG['REPETITIONS']} | Active Seed: {CURRENT_REP_SEED}")
                
            # Ensures all optimizers process the exact same datasets during the exact same repetition number.
            # Using 10,000 to match the size of standard FMNIST test set
            X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = train_test_split(
                X_raw, y_raw, test_size=10000, random_state=CURRENT_REP_SEED
            )
            
            # FMNIST doesn't need StandardScaler because we already normalized by / 255.0 globally
            X_train_g = X_tr_raw
            X_test_g = X_te_raw
            y_train_g = y_tr_raw
            y_test_g = y_te_raw
            
            best_score = setup["runner"]()
            results[method_name].append(best_score)
            REPETITION += 1

            # Secondary deep memory cleanup
            tf.keras.backend.clear_session()
            gc.collect()

    print("\n===== FINAL RESULTS ACROSS ALL REPETITIONS (Validation Accuracy) =====")
    for method_name, scores in results.items():
        calculate_stats(scores, method_name)
    
    # if ALL_METHODS_HISTORY:
    #     save_pbt_data(ALL_METHODS_HISTORY, filename="fmnist_results.pkl")
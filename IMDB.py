# -*- coding: utf-8 -*-
"""
Modified for IMDB Dataset (NLP Binary Classification)
Includes: Global History Tracking and High-Performance Custom Training Loops via AutoGraph.
"""
# !pip install ax-platform pyyaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces TensorFlow to ignore the GPU for faster CPU execution
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
import forced_walk

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    "TRIALS": 60,
    "REPETITIONS": 10,
    "TARGET_METRIC_NAME": "binary_crossentropy_loss", 
    "MAX_WORDS": 10000,
    "MAX_LEN": 256
}
print("IMDB Dynamic PARAMS (NLP Architecture + High-Performance AutoGraph Loop)")

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
# Define the active search space here. 
# To disable tuning for a specific parameter, simply comment it out in HP_SPACE.
# The objective functions will automatically fall back to the static value defined in DEFAULT_HPARAMS.
HP_SPACE = {
    "batch_size": {"type": "int", "low": 16, "high": 1024},
    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.005, "log": True},
    "epochs": {"type": "int", "low": 5, "high": 20},
    "activation": {"type": "categorical", "choices": ["sigmoid", "relu", "tanh"]},
    "optimizer_name": {"type": "categorical", "choices": ["Adam", "RMSprop", "SGD"]},
    "embedding_dim": {"type": "int", "low": 32, "high": 128}
}

DEFAULT_HPARAMS = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10,
    "activation": "relu",
    "optimizer_name": "Adam",
    "embedding_dim": 64
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
print("Loading IMDB Dataset into memory...")

# Keep the original IMDB train/test split 
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = imdb.load_data(
    num_words=CONFIG["MAX_WORDS"]
)

# Pad the official IMDB train/test sets separately
X_train_base = pad_sequences(
    X_train_raw,
    maxlen=CONFIG["MAX_LEN"]
)

X_test_base = pad_sequences(
    X_test_raw,
    maxlen=CONFIG["MAX_LEN"]
)

y_train_base = y_train_raw.reshape(-1, 1).astype(np.float32)
y_test_base = y_test_raw.reshape(-1, 1).astype(np.float32)

print(f"IMDB training samples: {len(X_train_base)}")
print(f"IMDB test samples:     {len(X_test_base)}")

# --------------------------
# Model builder
# --------------------------
def build_model(learning_rate, activation, optimizer_name, embedding_dim, rep_seed):
    """Constructs the Keras model and explicitly returns the model and optimizer for custom training."""
    
    kernel_init = keras.initializers.GlorotUniform(seed=rep_seed) if rep_seed is not None else "glorot_uniform"
    embed_init = keras.initializers.RandomUniform(seed=rep_seed) if rep_seed is not None else "uniform"
    
    model = keras.Sequential([
        keras.layers.Embedding(CONFIG["MAX_WORDS"], int(embedding_dim), input_length=CONFIG["MAX_LEN"], embeddings_initializer=embed_init),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64, activation=activation, kernel_initializer=kernel_init),
        keras.layers.Dense(1, activation='sigmoid', kernel_initializer=kernel_init)     
    ])
    
    if optimizer_name == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        
    return model, optimizer

def train_evaluate(batch_size, learning_rate, epochs, activation, optimizer_name, embedding_dim, X_tr, y_tr, X_te, y_te):
    """Executes a highly optimized custom training loop using AutoGraph and tf.data."""
    model, optimizer = build_model(
        learning_rate, activation, optimizer_name, embedding_dim, rep_seed=CURRENT_REP_SEED
    )
    
    loss_fn = keras.losses.BinaryCrossentropy()
    
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
        return loss_fn(y, predictions)

    avg_val_loss = float('inf')

    for epoch in range(int(epochs)):
        # Training loop
        for x_batch, y_batch in train_dataset:
            train_loss = train_step(x_batch, y_batch)
            
            # Immediately catch diverging gradients and prune the trial
            if tf.math.is_nan(train_loss):
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                return float('nan')
        
        # Validation loop
        val_loss_sum = 0.0
        val_batches = 0
        for x_batch_val, y_batch_val in val_dataset:
            val_loss_sum += val_step(x_batch_val, y_batch_val)
            val_batches += 1
            
        # Overwrite the validation loss at each epoch
        avg_val_loss = float(val_loss_sum / val_batches)
        
    # Aggressive memory cleanup for the custom loop
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Return the exact loss at the final epoch proposed by the optimizer
    return avg_val_loss


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
    study = forced_walk.create_fw_study(direction="minimize", terminate_value=None)
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
    study = optuna.create_study(direction="minimize", sampler=sampler)
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
    study = optuna.create_study(direction="minimize", sampler=sampler)
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
        minimize=True,
        random_seed=ax_seed
    )

    df = experiment.fetch_data().df
    best_row = df.sort_values("mean").iloc[0]
    return best_row["mean"]


# --------------------------
# Print Stats Helper
# --------------------------
def calculate_stats(arr, name):
    if not arr:
        return
    np_arr = np.array(arr)
    # Remove NaN values caused by divergent trials before calculating statistics
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
            # Use the official IMDB train/test split, exactly as in CODEB.
            # Do NOT combine the original train and test sets.
            X_train_g = X_train_base
            X_test_g = X_test_base
            y_train_g = y_train_base
            y_test_g = y_test_base
            
            best_score = setup["runner"]()
            results[method_name].append(best_score)
            REPETITION += 1

            # Secondary deep memory cleanup
            tf.keras.backend.clear_session()
            gc.collect()

    print("\n===== FINAL RESULTS ACROSS ALL REPETITIONS (Validation Binary Crossentropy Loss) =====")
    for method_name, scores in results.items():
        calculate_stats(scores, method_name)
    
    # if ALL_METHODS_HISTORY:
    #     save_pbt_data(ALL_METHODS_HISTORY, filename="imdb_results.pkl")
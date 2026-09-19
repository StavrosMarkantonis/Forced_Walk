# -*- coding: utf-8 -*-
import numpy as np
import csv
import pickle
import os
import logging

import optuna
from optuna.samplers import RandomSampler
from ax.service.managed_loop import optimize
from ax.utils.common.logger import get_logger

import forced_walk as fw

# To achieve a mathematically rigorous paired experiment across completely different hyperparameter optimization frameworks (Optuna, Ax, and Forced Walk), sharing the global seed is not enough.

# Even if you pass seed=42 to all of them, they will generate different initial hyperparameter guesses because they use different underlying mathematical sequences (e.g., Ax uses Sobol sequences, while Optuna TPE uses a standard random sampler for its startup trials).

# ==========================================
# 1. MATHEMATICAL TEST FUNCTIONS
# ==========================================


def schwefel(param):
    x = np.array(param)
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def rastrigin(param):
    x = np.array(param)
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(param):
    x = np.array(param)
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20 + np.e

def zakharov(param):
    x = np.array(param)
    n = len(x)
    sum_term_1 = np.sum(x**2)
    sum_term_2 = np.sum(0.5 * np.arange(1, n + 1) * x)
    return sum_term_1 + sum_term_2**2 + sum_term_2**4

def levy(param):
    x = np.array(param)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def rosenbrock(param):
    x = np.array(param)
    if len(x) < 2:
        return (x[0] - 1)**2
    sum_term = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
    return sum_term

def michalewicz(param, m=10):
    x = np.array(param)
    n = len(x)
    sum_term = np.sum(np.sin(x) * (np.sin(np.arange(1, n + 1) * x**2 / np.pi))**(2 * m))
    return -sum_term

def branin(param, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, d=6, e=10, f=1/(8*np.pi)):
    x = np.array(param)
    if len(x) != 2:
        raise ValueError("Branin function requires exactly 2 dimensions.")        
    x1, x2 = x[0], x[1]    
    term1 = a * (x2 - b * x1**2 + c * x1 - d)**2
    term2 = e * (1 - f) * np.cos(x1)    
    return term1 + term2 + e

def hartmann3(param):
    x = np.array(param)
    alpha = np.array([1, 1.2, 3, 3.2])
    A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
    P = np.array([[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.0381, 0.5743, 0.8828]])
    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(3):
            inner += A[ii, jj] * ((x[jj] - P[ii, jj]) ** 2)
        outer += alpha[ii] * np.exp(-inner)
    return -outer

def hartmann6C(param):
    x = np.array(param).flatten()
    if x.shape != (6,):
        raise ValueError("Hartmann 6D function requires exactly 6 dimensions.")
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 10**-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    exponent = np.sum(A * (x - P)**2, axis=1)    
    return -np.sum(alpha * np.exp(-exponent))

def sixHump(param):
    x = np.array(param)
    if len(x) != 2:
        raise ValueError("Six-Hump Camel function requires exactly 2 dimensions.")
    x1, x2 = x[0], x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3

def sinOne(param):
    x = param[0]
    return np.sin(x) + np.sin(10 * x / 3)

def goldsteinPrice(param):
    x = np.array(param)
    if len(x) != 2:
        raise ValueError("Goldstein-Price function requires exactly 2 dimensions.")
    x1, x2 = x[0], x[1]
    a = x1 + x2 + 1.0
    b = 19.0 - 14.0 * x1 + 3.0 * x1 * x1 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2 * x2
    c = 2.0 * x1 - 3.0 * x2
    d = 18.0 - 32.0 * x1 + 12.0 * x1 * x1 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2 * x2
    f = ( 1.0 + a * a * b ) * ( 30.0 + c * c * d )    
    return f


# ==========================================
# 2. BENCHMARK CONFIGURATION REGISTRY
# ==========================================
BENCHMARK_REGISTRY = {
    "zakharov": {"func": zakharov, "bounds": [("paramA", -10, 10), ("paramB", -10, 10), ("paramC", -10, 10), ("paramD", -10, 10)]},
    "michalewicz": {"func": michalewicz, "bounds": [("paramA", 0, 3.14), ("paramB", 0, 3.14), ("paramC", 0, 3.14), ("paramD", 0, 3.14)]},
    "rosenbrock": {"func": rosenbrock, "bounds": [("paramA", -5, 10), ("paramB", -5, 10), ("paramC", -5, 10), ("paramD", -5, 10)]},
    "schwefel": {"func": schwefel, "bounds": [("paramA", -500, 500), ("paramB", -500, 500), ("paramC", -500, 500), ("paramD", -500, 500)]},
    "ackley": {"func": ackley, "bounds": [("paramA", -32.768, 32.768), ("paramB", -32.768, 32.768), ("paramC", -32.768, 32.768), ("paramD", -32.768, 32.768)]},
    "rastrigin": {"func": rastrigin, "bounds": [("paramA", -5.12, 5.12), ("paramB", -5.12, 5.12), ("paramC", -5.12, 5.12), ("paramD", -5.12, 5.12)]},
    "levy": {"func": levy, "bounds": [("paramA", -10, 10), ("paramB", -10, 10), ("paramC", -10, 10), ("paramD", -10, 10)]},
    "branin": {"func": branin, "bounds": [("paramA", -5, 10), ("paramB", 0, 15)]},
    "hartmann3": {"func": hartmann3, "bounds": [("paramA", 0, 1), ("paramB", 0, 1), ("paramC", 0, 1)]},
    "hartmann6": {"func": hartmann6C, "bounds": [("paramA", 0, 1), ("paramB", 0, 1), ("paramC", 0, 1), ("paramD", 0, 1), ("paramE", 0, 1), ("paramF", 0, 1)]},
    "sixHump": {"func": sixHump, "bounds": [("paramA", -3, 3), ("paramB", -2, 2)]},
    "sinOne": {"func": sinOne, "bounds": [("paramA", -10, 10)]},
    "goldsteinPrice": {"func": goldsteinPrice, "bounds": [("paramA", -2, 2), ("paramB", -2, 2)]},
}


# ==========================================
# 3. OPTIMIZATION RUNNERS
# ==========================================

def run_ax_trial(func_name, num_trials):
    target_func = BENCHMARK_REGISTRY[func_name]["func"]
    bounds = BENCHMARK_REGISTRY[func_name]["bounds"]

    ax_parameters = [
        {"name": b[0], "type": "range", "bounds": [float(b[1]), float(b[2])], "value_type": "float"}
        for b in bounds
    ]

    def ax_objective(parameters):
        params = [parameters[b[0]] for b in bounds]
        jitter = np.random.randn() * 1e-5
        return target_func(params) + jitter

    best_parameters, best_values, experiment, model = optimize(
        parameters=ax_parameters,
        evaluation_function=ax_objective,
        objective_name="objective",
        total_trials=num_trials,
        minimize=True
    )

    df = experiment.fetch_data().df
    return df.sort_values("mean").iloc[0]["mean"]

def run_forced_walk(func_name, num_trials):
    target_func = BENCHMARK_REGISTRY[func_name]["func"]
    bounds = BENCHMARK_REGISTRY[func_name]["bounds"]
    
    def fw_objective(trial):
        params = [trial.suggest_float(b[0], b[1], b[2]) for b in bounds]
        return target_func(params)

    study = fw.create_fw_study(direction="minimize", terminate_value=-float('inf'))
    study.optimize(fw_objective, n_trials=num_trials)
    
    return study.best_value

def run_optuna_trial(func_name, num_trials, sampler_type="tpe"):
    target_func = BENCHMARK_REGISTRY[func_name]["func"]
    bounds = BENCHMARK_REGISTRY[func_name]["bounds"]
    
    def objective(trial):
        params = [trial.suggest_float(b[0], b[1], b[2]) for b in bounds]
        return target_func(params)

    sampler = optuna.samplers.TPESampler() if sampler_type == "tpe" else RandomSampler()
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    study.optimize(objective, n_trials=num_trials)
    return study.best_value

# Wrappers to fit the generic runner signature
def run_optuna_tpe(func_name, num_trials):
    return run_optuna_trial(func_name, num_trials, sampler_type="tpe")

def run_random_search(func_name, num_trials):
    return run_optuna_trial(func_name, num_trials, sampler_type="random")

# Mapping of method names to their runner functions
METHOD_REGISTRY = {
    "AX": run_ax_trial,
    "Optuna_TPE": run_optuna_tpe,
    "Random_Search": run_random_search,
    "Forced_Walk": run_forced_walk
}


# ==========================================
# 4. BENCHMARKING PIPELINE & EXPORT
# ==========================================

def calculate_stats(arr):
    np_arr = np.array(arr)
    return np.mean(np_arr), np.std(np_arr, ddof=1)

def run_benchmark_suite(methods_to_test, functions_to_test, runs=20, trials_per_run=200, temp_folder="Synthetic_Functions", excel_filename="function_analysis_results.csv"):
    os.makedirs(temp_folder, exist_ok=True)
    
    # Safely join the folder path and the user-defined excel filename
    output_csv = os.path.join(temp_folder, excel_filename)
    
    # Write CSV header if file doesn't exist
    if not os.path.isfile(output_csv):
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Method", "Function", "Mean", "StdDev", "Raw Values"])
            
    summary_stats = []
    
    for method_name in methods_to_test:
        if method_name not in METHOD_REGISTRY:
            print(f"⚠️ Warning: {method_name} not found in method registry. Skipping.")
            continue
            
        runner_func = METHOD_REGISTRY[method_name]
        
        for func_name in functions_to_test:
            if func_name not in BENCHMARK_REGISTRY:
                print(f"⚠️ Warning: {func_name} not found in function registry. Skipping.")
                continue
                
            print(f"\n" + "="*50)
            print(f"Testing Method: {method_name} | Function: {func_name}")
            print("="*50)
            
            # Execute runs
            run_values = [runner_func(func_name, trials_per_run) for _ in range(runs)]
            
            # Calculate stats
            mean_val, std_val = calculate_stats(run_values)
            summary_stats.append((method_name, func_name, mean_val, std_val))
            
            # Append to CSV
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([method_name, func_name, f"{mean_val:.6f}", f"{std_val:.6f}", str(run_values)])
                
    # Display final summary table
    print("\n\n===== BENCHMARK SUMMARY (Mean ± Std) =====")
    for method, func, mean, std in summary_stats:
        print(f"[{method}] {func:<15} -> Mean: {mean:.6f}, Std: {std:.6f}")

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("ax").setLevel(logging.CRITICAL)
    
    # ---------------------------------------------------------
    # USER CONFIGURATION 
    # ---------------------------------------------------------
    
    # 1. Select the methods to evaluate
    # Options: "AX", "Optuna_TPE", "Random_Search", "Forced_Walk"
    selected_methods = ["Forced_Walk"]
    
    # 2. Select the mathematical functions to benchmark against
    # Options: "branin", "hartmann6", "sixHump", etc.
    selected_functions = ["branin"]  
    
    # Execute the suite
    run_benchmark_suite(
        methods_to_test=selected_methods, 
        functions_to_test=selected_functions, 
        runs=20, 
        trials_per_run=200, 
        temp_folder="Synthetic_Functions",
        excel_filename="function_analysis_results.csv"  
    )
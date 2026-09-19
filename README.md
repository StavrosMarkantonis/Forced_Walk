# Forced Walk Optimization Framework

**Forced Walk** is a scalable engine for black-box optimization specifically engineered for dynamic applications such as the hyperparameter tuning of Reinforcement Learning frameworks. It turns the challenge of finding global optima into an agent-based exploration task, allowing it to navigate complex, "jittery" parameter spaces.

Traditional Bayesian solvers (like those using Gaussian Processes) suffer from a massive **cubic bottleneck**, $\mathcal{O}(N^3)$. By using a **Neural Policy surrogate** instead of a GP, Forced Walk scales **linearly**. This means you get consistent performance even as your search history grows, making it ideal for high-dimensional tuning.

Unlike static optimizers, this framework was specifically built to handle the "moving targets" found in Reinforcement Learning. 
* **Noise-Resistant:** The deep learning backbone naturally filters out stochastic noise in reward signals.
* **Adaptive:** It is optimized for non-stationary landscapes (like hyperparameter scheduling) where data distributions shift over time.

## Algorithm Mechanics

<img width="1800" height="686" alt="method" src="https://github.com/user-attachments/assets/fb225376-c1ab-4bfa-b078-4d5dc47d7895" />

The engine utilizes a **dual-phase search strategy** designed to optimize black-box objective functions without requiring explicit gradients. It operates by balancing raw exploration with a neural-network-backed selection process.

### 1. Dynamic Scaling (The Growth Curve)
Instead of a fixed sample size, the framework initializes with a lean candidate pool. As the model gathers more environment interactions, the population size expands following a **logistic growth pattern**. This ensures computational efficiency in early training while providing the density needed for fine-tuned convergence in later stages.


### 2. Hierarchical Filtering (The Selection Pipeline)
The search trajectory is managed via a two-stage filter:

* **Macro-Sampling:** Generates a wide-reaching set of seeds around the current best-known parameters to avoid local optima.
* **Micro-Refinement:** A trained **Value Network** acts as a surrogate evaluator, identifying high-potential seeds. The system then "zooms in" to create dense local clusters around these points, effectively approximating a policy gradient without backpropagation through the environment.

### 3. Optimization & Memory Management
* **Winner-Take-All Evaluation:** Only the top candidate from each cluster is sent to the expensive objective function. This drastically reduces the number of "real" environment steps required.
* **Buffer Pruning:** To handle non-stationary data, the system employs a **temporal sliding window**. This automatically discards stale experience that no longer reflects the current policy's search space, preventing the Value Network from over-fitting to outdated trajectories.
* **Auto-Zoom:** If the reward signal plateaus (stagnation), the framework automatically constricts the exploration radius to focus the search on the immediate vicinity of the current peak.

---

## Requirements

Ensure the following dependencies are installed in your Python environment:
* `numpy`
* `scikit-learn`
* `tensorflow=2.10.0` ---Forced Walk code is optimized to use this version of TensorFlow---

---


## Usage Guide

The Forced Walk framework utilizes a "Define-by-Run" API, meaning you define your parameter search space dynamically inside the objective function itself.

### Example 1: Simple Use Case

To use the algorithm with its default configuration, simply define your objective function with the `trial.suggest_*` methods, and pass it to the optimization study.

```python
import forced_walk

def objective(trial):
    # 1. Define the search space dynamically
    batch_size = trial.suggest_int("batch_size", 16, 1024)
    learning_rate = trial.suggest_float("learning_rate", 0.0005, 0.002)
    epochs = trial.suggest_int("epochs", 10, 40)
    activation = trial.suggest_categorical("activation", ["sigmoid", "relu", "tanh"])
    
    # 2. Evaluate your black-box model (e.g., training a machine learning model)
    score = my_black_box_model_train_and_evaluate(
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        epochs=epochs, 
        activation=activation
    )
    
    return score

# Create the study and run the optimization
study = forced_walk.create_fw_study(direction="minimize")

print("Starting Forced Walk Optimization...")
study.optimize(objective, n_trials=100)

print(f"Best Score Achieved: {study.best_value}")
```

### Example 2: Early Termination for Target Objectives (RL Context)

In many scenarios, such as Reinforcement Learning environments, you may not need to run the algorithm until the evaluation budget is entirely exhausted. Instead, the environment is considered "solved" once the agent achieves a specific target reward (e.g., reaching a score of 200). 

You can use the `terminate_value` parameter to instantly halt the algorithm as soon as a candidate configuration meets or surpasses this threshold. This prevents unnecessary evaluations and saves significant computational resources.

```python
import forced_walk

def rl_objective(trial):
    # 1. Define the policy hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    
    # 2. Train the RL agent and return the episodic reward
    reward = train_rl_agent(learning_rate, gamma, batch_size)
    
    return reward

# Create a study aiming to MAXIMIZE the reward.
# The algorithm will terminate early if any trial returns 200.0 or higher.
study = forced_walk.create_fw_study(
    direction="maximize", 
    terminate_value=200.0
)

print("Starting optimization. Will terminate early if target reward is reached...")
# Even if n_trials is 500, it will stop at trial 42 if the reward hits 200.0
study.optimize(rl_objective, n_trials=500)

print(f"Optimization finished! Best Reward: {study.best_value}")
```

### Example 3: Hartmann 6 optimization
Example code for finding the global minimum of the 6-dimensional Hartmann equation (https://www.sfu.ca/~ssurjano/hart6.html).
```python
import numpy as np
import forced_walk

def hartmann6(param):
    """
    Hartmann 6-Dimensional function
    x must be a NumPy array of shape (6,)
    Global Minimum: approximately f(x)≈−3.322 at (0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054)
    """
    x = np.array(param).flatten()
    if x.shape != (6,):
        raise ValueError("Hartmann 6D function requires exactly 6 dimensions.")
    # Standard parameters
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
    
    # Calculate the inner sum: sum(A_ij * (x_j - P_ij)^2) for all i
    exponent = np.sum(A * (x - P)**2, axis=1)    
    # Calculate the final function value
    return -np.sum(alpha * np.exp(-exponent))

# --------------------------
# Forced Walk Objective
# --------------------------
def optimize_function(trial):   
    paramA = trial.suggest_float("paramA", 0, 1)
    paramB = trial.suggest_float("paramB", 0, 1)
    paramC = trial.suggest_float("paramC", 0, 1)
    paramD = trial.suggest_float("paramD", 0, 1)
    paramE = trial.suggest_float("paramE", 0, 1)
    paramF = trial.suggest_float("paramF", 0, 1)

    score = hartmann6([paramA, paramB ,paramC, paramD, paramE, paramF])
    return score
    
study = forced_walk.create_fw_study(direction="minimize")
study.optimize(optimize_function, n_trials=200)   
print(study.best_value)
```


## Configuration Guide

### Forced Walk Configuration Parameters

The following internal hyperparameters govern the behavior of the Forced Walk algorithm. They are defined and can be customized within the `config.yaml` file.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`logging`** | String | Enables console logging of trial evaluations and discovered best values (`"True"` or `"False"`). |
| **`use_colors`** | String | Deactivate the console's colored output (`"True"` or `"False"`). |
| **`base_scale`** | Int | Search grid resolution. Serves as the fundamental denominator for mapping discrete stochastic steps into continuous parameter spaces. Higher values yield finer minimum step sizes. |
| **`search_radius`** | Float | Global exploration radius multiplier ($\delta$). Must be in the range `(0, 0.5]`. Defines the maximum span of randomized steps relative to the base scale. A value of `0.5` establishes a full-space diameter of 1.0, covering 100% of the parameter bounds. |
| **`beta`** | Int | Phase 1 survival count ($\beta$), or beam width. Determines the number of top-performing candidate points retained after the global surrogate filtering step to serve as pivot points for local search. |
| **`tau`** | Int | Stagnation threshold ($\tau$). The number of consecutive trial evaluations without discovering a new global best before adaptive step-scaling (zooming) is triggered. |
| **`zeta`** | Float | Contraction factor ($\zeta$). The multiplier applied to systematically constrict the search resolution (shrinking the trust region) when the stagnation threshold is reached. |
| **`max_zoom`** | Int | Maximum allowable zoom magnification. Prevents the search radius from scaling down into mathematical collapse or infinitesimally small step sizes after repeated constrictions. |
| **`mu`** | Float | Forgetting factor ($\mu$). Must be in the range `[0, 1)`. The fraction of the oldest historical trial data to permanently discard before training the surrogate model (e.g., `0.3` drops the oldest 30%), forcing the network to overfit to the local topography. |
| **`max_samples`** | Int/Str | History truncation limit. The maximum number of recent trials to include in the surrogate model's training set. Set to `"False"` for unbounded memory. |
| **`rho`** | Int | Warm-up period ($\rho$). Must be strictly `> 2`. The number of initial, purely random evaluations used to populate the historical data buffer before surrogate-guided search begins, preventing cold-start bias. |
| **`R_local`** | Int | Local sampling intensity. The number of dense candidate mutations generated around each surviving pivot point during the Phase 2 proximal refinement stage. |
| **`lambda`** | Float | Locality factor multiplier ($\lambda$). Must be in the range `(0, 1]`. Dictates the local refinement radius strictly as a fraction of the current global search radius ($\delta_{\text{local}} = \lambda \cdot \delta$). |

### Surrogate Neural Network Hyperparameters

The underlying neural network guiding the surrogate filtering can also be fully customized via the same `config.yaml` file.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`nn_activation`** | String | Activation function for the surrogate model's hidden layers (e.g., `"relu"`, `"tanh"`). |
| **`nn_layers`** | Int | Number of hidden dense layers in the surrogate neural network. |
| **`nn_nodes`** | Int | Number of neurons per hidden layer. |
| **`nn_learning_rate`** | Float | Learning rate for the Adam optimizer during surrogate model training. |
| **`nn_batch_size`** | Int | Mini-batch size used when fitting the surrogate model. |
| **`early_stopping`** | String | Toggles early stopping (`"True"` or `"False"`). If enabled, training halts if the loss does not improve for 10 epochs, restoring the best weights. |
| **`nn_epochs_early`** | Int | Number of training epochs to execute when the historical buffer contains fewer than 1,000 data points. |
| **`nn_epochs_late`** | Int | Number of training epochs to execute when the historical buffer contains 1,000 or more data points. |
| **`force_cpu`** | Bool | Forces TensorFlow to execute on the CPU. Recommended to avoid GPU memory transfer overhead and latency when frequently retraining very small networks. |
| **`random_seed`** | Int | Global random seed to ensure search initialization reproducibility. |


## Reproducing the Results

### Section IV-B. PHASE I: SYNTHETIC MATHEMATICAL EXPRESSIONS & Section IV-C. ABLATION STUDY
Execute the `synthetic_functions.py` script to reproduce the experiments evaluating the synthetic benchmark functions.

### Section IV-D. PHASE I: SEQUENTIAL HPO IN SUPERVISED LEARNING
- Execute `california.py` to reproduce the experiments using the California Housing dataset.
- Execute `adult_income.py` to reproduce the experiments using the Adult Census Income dataset.
- Execute `IMDB.py` to reproduce the experiments using the IMDB dataset.
- Execute `FMNIST.py` to reproduce the experiments using the FMNIST dataset.

### Section IV-G. PHASE II: ONLINE HPO IN RL
- Execute `cartpole_FW_PBT.py` to reproduce the experiments evaluating the FW and standard PBT methods in the `CartPole-v1` environment.
- Execute `cartpole_PB2.py` to reproduce the experiments evaluating the PB2 method in the `CartPole-v1` environment.
- Execute `lunarlander_FW_PBT.py` to reproduce the experiments evaluating the FW and standard PBT methods in the `LunarLander-v3` environment.
- Execute `lunarlander_PB2.py` to reproduce the experiments evaluating the PB2 method in the `LunarLander-v3` environment.


## Experimental Data

### Section IV-B. PHASE I: SYNTHETIC MATHEMATICAL EXPRESSIONS
The raw experimental results are available in the Excel file `raw_data_sectionsB-C-D-G.xlsx`.

### Section IV-C. ABLATION STUDY
The raw experimental results are available in the Excel file `raw_data_sectionsB-C-D-G.xlsx`.

### Section IV-D. PHASE I: SEQUENTIAL HPO IN SUPERVISED LEARNING
The raw experimental results are available in the Excel file `raw_data_sectionsB-C-D-G.xlsx`. Additionally, the data used to plot the optimization trajectories and hyperparameter evolution across the four supervised learning benchmarks are stored as `.pkl` files in the `SectionIV-D_7_param_diagram_data` directory.

### Section IV-E. PHASE II: ONLINE HPO IN SUPERVISED LEARNING
The raw experimental results are provided as `.pkl` files in the `SectionIV-E_raw_data` directory.

### Section IV-F. PHASE II: ONLINE HPO IN SELF-PLAY RL
The raw experimental results are provided as `.pkl` files in the `SectionIV-F-diagrams_data_backgammon` directory.

### Section IV-G. PHASE II: ONLINE HPO IN RL
The summarized experimental results are available in the Excel file `raw_data_sectionsB-C-D-G.xlsx`. Detailed analytical results for each individual experiment are located in the `SectionIV-G_raw_data` directory.


# Forced Walk Optimization Framework

Forced Walk is a high-performance, black-box optimization framework reformulated as a Reinforcement Learning (RL) process. It is specifically designed to navigate complex, non-differentiable parameter spaces efficiently. 

Instead of relying on Gaussian Processes, which suffer from cubic computational complexity, Forced Walk utilizes a Neural Network as a high-fidelity surrogate policy. This approach enables linear scaling and provides greater robustness to noise in stochastic environments.

<img width="1800" height="686" alt="method" src="https://github.com/user-attachments/assets/fb225376-c1ab-4bfa-b078-4d5dc47d7895" />


## Algorithm Mechanics

The framework explicitly decouples search dynamics: stochastic perturbation drives exploration, while policy-driven selection enforces exploitation. The search trajectory is guided by a Bilevel Filtering Policy:

1. **Global Exploration:** A candidate pool is sampled around the best-known solution. The size of this pool dynamically scales based on a Sigmoidal growth curve dependent on cumulative experience.
2. **Proximal Refinement:** The Value Network selects the top high-potential seeds. High-density clusters are generated locally around these seeds to simulate a local policy gradient.
3. **Greedy Selection & Experience Replay:** The single optimal candidate from each cluster is evaluated by the true black-box objective function. The state-reward tuples are saved to an Experience Replay Buffer to train the Value Network off-policy.
4. **Adaptive Step-Scaling & Dynamic Maintenance:** Upon detecting stagnation, the exploration radius is constricted by a zoom factor. A sliding window prunes older distal experiences to mitigate distribution shift during training.

---

## Requirements

Ensure the following dependencies are installed in your Python environment:
* `numpy`
* `scikit-learn`
* `tensorflow` (CPU execution is utilized internally to avoid GPU memory fragmentation during frequent micro-retrains)

---

## Usage Guide

The Forced Walk framework utilizes a "Define-by-Run" API, meaning you define your parameter search space dynamically inside the objective function itself.

### Example 1: Simple Use Case

To use the algorithm with its default configuration, simply define your objective function with the `trial.suggest_*` methods, and pass it to the optimization study.

```python
import forcedWalk

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
study = forcedWalk.create_fw_study(direction="minimize")

print("Starting Forced Walk Optimization...")
study.optimize(objective, n_trials=100)

print(f"Best Score Achieved: {study.best_value}")
```

### Example 2: Changing the Algorithm's Internal Parameters

If you need to tune the internal behavior of the Forced Walk algorithm (e.g., adjusting the stagnation limits or sampling intensity), you can pass a `forced_walk_parameters` dictionary when creating the study.

You only need to include the specific parameters you wish to override; the rest will fall back to their default values.

```python
import forcedWalk

# Define the specific internal parameters you want to alter for the algorithm
forced_walk_parameters = {
    "tau": 15,           # Lower the stagnation limit to zoom in faster
    "mu": 0.3,           # Set the sliding window to drop the oldest 30% of data
    "R_local": 5000      # Halve the local sampling intensity for faster execution
}

# Pass the dictionary into the study creation via the hyperparams argument
study = forcedWalk.create_fw_study(
    direction="minimize", 
    terminate_value=0.0000001,
    hyperparams=forced_walk_parameters
)

# Run the optimization
study.optimize(objective, n_trials=150)

print(f"Best Score Achieved: {study.best_value}")
```

### Example 3: Early Termination for Target Objectives (RL Context)

In many scenarios, such as Reinforcement Learning environments, you may not need to run the algorithm until the evaluation budget is entirely exhausted. Instead, the environment is considered "solved" once the agent achieves a specific target reward (e.g., reaching a score of 200). 

You can use the `terminate_value` parameter to instantly halt the algorithm as soon as a candidate configuration meets or surpasses this threshold. This prevents unnecessary evaluations and saves significant computational resources.

```python
import forcedWalk

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
study = forcedWalk.create_fw_study(
    direction="maximize", 
    terminate_value=200.0
)

print("Starting optimization. Will terminate early if target reward is reached...")
# Even if n_trials is 500, it will stop at trial 42 if the reward hits 200.0
study.optimize(rl_objective, n_trials=500)

print(f"Optimization finished! Best Reward: {study.best_value}")
```

### Example 4: Hartmann 6 optimization
Example code for finding the minima of the 6-dimensional Hartmann equation (https://www.sfu.ca/~ssurjano/hart6.html).
```python
import numpy as np
import forcedWalk

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
    
study = forcedWalk.create_fw_study(direction="minimize", hyperparams={
        "tau": 20,          # Set the stagnation limit
        "mu": 0,            # Set the sliding window
        "zeta": 2,          # Zoom factor used to constrict the exploration trust region
        "max_zoom": 48      # Maximum allowed constriction of the exploration trust region
    })
study.optimize(optimize_function, n_trials=200)   
print(study.best_value)
```

## Internal Parameter Configuration Guide

The following internal parameters govern the exploration/exploitation balance of the algorithm. They can be overridden using the dictionary method shown in Example 2.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`upper_search`** | Int | Global search radius boundary (percentage of space). For example, `50` searches the entire parameter space. |
| **`beta`** | Int | Evaluation batch size and selection bottleneck. Determines how many seeds survive Global Exploration. |
| **`tau`** | Int | Stagnation limit. Number of consecutive episodes without improvement before triggering adaptive step-scaling (zoom). |
| **`zeta`** | Float | Zoom factor. The multiplier used to constrict the exploration trust region upon stagnation. |
| **`mu`** | Float | Sliding window factor `[0, 1)`. Fraction of the oldest replay buffer experiences to prune (e.g., `0.3` drops oldest 30%). |
| **`max_zoom`** | Int | Maximum allowed zoom level. Prevents the search radius from scaling down into floating-point collapse. |
| **`rho`** | Int | Number of initial random samples for the stochastic warm-up phase to prevent cold-start bias. |
| **`R_local`** | Int | Local sampling intensity. Number of candidates generated in the proximal refinement clusters. |
| **`lambda`** | Float | Locality factor. The ratio linking the global exploration radius to the local refinement radius. |

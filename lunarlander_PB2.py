# -*- coding: utf-8 -*-

"""
LunarLander-v3
DQN + Population Based Bandits (PB2)

PB2 implementation based on the original PB2 formulation:
    Parker-Holder, Nguyen & Roberts,
    "Provably Efficient Online Hyperparameter Optimization
     with Population-Based Bandits", NeurIPS 2020.

The DQN implementation preserves:
    - Double DQN
    - Huber loss
    - replay buffer
    - Polyak/soft target updates

PB2 controls:
    - gamma
    - learning_rate
    - batch_size
    - epsilon
"""

# ============================================================
# 1. IMPORTS / THREAD CONFIGURATION
# ============================================================

import os
import yaml  # ADDED: to read from config.yaml

# THREADS = "4"

# os.environ["OMP_NUM_THREADS"] = THREADS
# os.environ["OPENBLAS_NUM_THREADS"] = THREADS
# os.environ["MKL_NUM_THREADS"] = THREADS
# os.environ["VECLIB_MAXIMUM_THREADS"] = THREADS
# os.environ["NUMEXPR_NUM_THREADS"] = THREADS
# os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"  # Forces aggressive glibc garbage collection on Linux

import csv
import copy
import gc
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import tensorflow as tf

# tf.config.threading.set_intra_op_parallelism_threads(int(THREADS))
# tf.config.threading.set_inter_op_parallelism_threads(int(THREADS))

import gymnasium as gym

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

from scipy.optimize import minimize


# ============================================================
# 2. GLOBAL CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

ENV_NAME = "LunarLander-v3"

STATE_DIM = 8
ACTION_DIM = 4

# ------------------------------------------------------------
# PB2 / PBT configuration
# ------------------------------------------------------------

NUM_ITERATIONS = 7  # Number of full independent PB2 runs
POPULATION_SIZE = 8

EPISODES_PER_GEN = 20
TOTAL_EPISODE_BUDGET = 2000

GENERATIONS = TOTAL_EPISODE_BUDGET // POPULATION_SIZE

TOP_K = 3

# LunarLander standard "solved" threshold
STOP_THRESHOLD = 200.0

# Evaluation
EVAL_EPISODES = 5

# ------------------------------------------------------------
# DQN
# ------------------------------------------------------------

BUFFER_SIZE = 40_000
MIN_REPLAY_SIZE = 1000
TRAIN_FREQ = 4
TAU = 0.005

# ------------------------------------------------------------
# PB2 hyperparameter bounds
# ------------------------------------------------------------

PB2_BOUNDS = {
    "gamma": (0.8, 0.999),
    "learning_rate": (0.0001, 0.01),
    "batch_size": (16, 1024),
    "epsilon": (0.01, 1.0),
}

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

CSV_FILENAME = "lunarlander_PB2.csv"

CSV_HEADER = [
    "experiment_id",
    "iteration",
    "generation",
    "total_population_episodes",
    "score",
    "learning_rate",
    "batch_size",
    "gamma",
    "epsilon",
    "wall_time",
    "episodes_to_solve"
]

LOG_EXECUTION_TIME = True
CSV_FILENAME_FORCED_EXECUTION_TIME = 'lunarlander_FW_Execution_Time.csv'

CSV_HEADER_TIME = [
    'experiment_id', 'run_iteration', 'gen', 'number_of_episodes', 'FW_iterationtime' 
]

# ============================================================
# 3. EXPERIMENT ID
# ============================================================

EXPERIMENT_ID = "PB2_LUNARLANDER_A01"


# ============================================================
# 4. DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(self, state_dim, action_dim, hparams, agent_seed=None):

        self.hparams = copy.deepcopy(hparams)
        self.tau = TAU

        # ----------------------------------------------------
        # Replay buffer
        # ----------------------------------------------------
        self.memory_size = BUFFER_SIZE

        self.states = np.zeros((self.memory_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.memory_size, dtype=np.int32)
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        self.next_states = np.zeros((self.memory_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.memory_size, dtype=np.float32)

        self.mem_ptr = 0
        self.mem_count = 0
        self.score = None

        # Local seed initializers to avoid tf.random.set_seed() global statements
        init1 = tf.keras.initializers.GlorotUniform(seed=agent_seed) if agent_seed else 'glorot_uniform'
        init2 = tf.keras.initializers.GlorotUniform(seed=agent_seed + 1 if agent_seed else None)
        init3 = tf.keras.initializers.GlorotUniform(seed=agent_seed + 2 if agent_seed else None)

        # ----------------------------------------------------
        # Q network
        # ----------------------------------------------------
        self.q_net = models.Sequential([
            layers.Input(shape=(state_dim,)),
            layers.Dense(128, activation="relu", kernel_initializer=init1),
            layers.Dense(128, activation="relu", kernel_initializer=init2),
            layers.Dense(action_dim, kernel_initializer=init3)
        ])

        # ----------------------------------------------------
        # Target network
        # ----------------------------------------------------
        self.target_net = tf.keras.models.clone_model(self.q_net)
        self.target_net.set_weights(self.q_net.get_weights())

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------
        self.optimizer = optimizers.Adam(learning_rate=self.hparams["learning_rate"])
        self.loss_fn = losses.Huber()

    # ========================================================
    # REPLAY BUFFER
    # ========================================================
    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_ptr

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.mem_ptr = (self.mem_ptr + 1) % self.memory_size
        self.mem_count = min(self.mem_count + 1, self.memory_size)

    def sample_memory(self, batch_size):
        actual_size = min(self.mem_count, int(batch_size))
        indices = np.random.choice(self.mem_count, actual_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    # ========================================================
    # DOUBLE DQN TRAINING
    # ========================================================
    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=(None, STATE_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, STATE_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        ]
    )
    def train_step(self, states, actions, rewards, next_states, dones, gamma):
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)

        next_q_online = self.q_net(next_states, training=False)
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
        next_q_target = self.target_net(next_states, training=False)

        next_action_indices = tf.stack([batch_indices, next_actions], axis=1)
        next_q = tf.gather_nd(next_q_target, next_action_indices)

        targets = rewards + gamma * next_q * (1.0 - dones)

        with tf.GradientTape() as tape:
            q_values = self.q_net(states, training=True)
            action_indices = tf.stack([batch_indices, actions], axis=1)
            q_action = tf.gather_nd(q_values, action_indices)
            loss = self.loss_fn(targets, q_action)

        gradients = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_net.trainable_variables))

        for target_var, local_var in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
            target_var.assign(self.tau * local_var + (1.0 - self.tau) * target_var)

    # ========================================================
    # ACTION
    # ========================================================
    @tf.function(
        reduce_retracing=True,
        input_signature=[tf.TensorSpec(shape=(1, STATE_DIM), dtype=tf.float32)]
    )
    def get_action(self, state):
        q_values = self.q_net(state, training=False)
        return tf.argmax(q_values[0])

    # ========================================================
    # HYPERPARAMETER UPDATE
    # ========================================================
    def update_hparams(self, new_hparams):
        self.hparams = copy.deepcopy(new_hparams)
        self.optimizer.learning_rate.assign(self.hparams["learning_rate"])

    # ========================================================
    # EXPLOIT
    # ========================================================
    def copy_weights_from(self, elite_agent):
        self.q_net.set_weights(elite_agent.q_net.get_weights())
        self.target_net.set_weights(elite_agent.target_net.get_weights())
        
        # Buffer copy logic removed so agents retain their own memories
        
        for variable in self.optimizer.variables():
            variable.assign(tf.zeros_like(variable))


# ============================================================
# 5. TRAIN ONE AGENT
# ============================================================

def train_agent(agent):
    env = gym.make(ENV_NAME)

    gamma = float(agent.hparams["gamma"])
    epsilon = float(agent.hparams["epsilon"])
    batch_size = int(agent.hparams["batch_size"])

    local_steps = 0

    for episode in range(EPISODES_PER_GEN):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            local_steps += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_input = np.asarray(state, dtype=np.float32)[None, :]
                action = int(agent.get_action(state_input).numpy())

            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, terminated)
            state = next_state

            if agent.mem_count >= MIN_REPLAY_SIZE and local_steps % TRAIN_FREQ == 0:
                states, actions, rewards, next_states, dones = agent.sample_memory(batch_size)
                agent.train_step(states, actions, rewards, next_states, dones, gamma)

    evaluation_rewards = []
    for _ in range(EVAL_EPISODES):
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            state_input = np.asarray(state, dtype=np.float32)[None, :]
            action = int(agent.get_action(state_input).numpy())
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        evaluation_rewards.append(episode_reward)

    env.close()
    return float(np.mean(evaluation_rewards))


# ============================================================
# 6. PB2 TIME-VARYING SQUARED EXPONENTIAL KERNEL
# ============================================================

class TV_SquaredExp(Kernel):

    def __init__(
        self,
        variance=1.0,
        lengthscale=1.0,
        epsilon=0.1,
        variance_bounds=(1e-5, 1e5),
        lengthscale_bounds=(1e-5, 1e5),
        epsilon_bounds=(1e-5, 0.5)
    ):
        self.variance = variance
        self.lengthscale = lengthscale
        self.epsilon = epsilon
        self.variance_bounds = variance_bounds
        self.lengthscale_bounds = lengthscale_bounds
        self.epsilon_bounds = epsilon_bounds

    @property
    def hyperparameter_variance(self):
        return Hyperparameter("variance", "numeric", self.variance_bounds)

    @property
    def hyperparameter_lengthscale(self):
        return Hyperparameter("lengthscale", "numeric", self.lengthscale_bounds)

    @property
    def hyperparameter_epsilon(self):
        return Hyperparameter("epsilon", "numeric", self.epsilon_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        Y = np.atleast_2d(Y)

        epsilon = np.clip(self.epsilon, 1e-5, 0.5)

        T1 = X[:, 0].reshape(-1, 1)
        T2 = Y[:, 0].reshape(-1, 1)
        time_distance = pairwise_distances(T1, T2, metric="cityblock")
        time_kernel = (1.0 - epsilon) ** (0.5 * time_distance)

        X_spatial = X[:, 1:]
        Y_spatial = Y[:, 1:]
        spatial_distance = euclidean_distances(X_spatial, Y_spatial)
        spatial_kernel = self.variance * np.exp(-np.square(spatial_distance) / self.lengthscale)

        K = spatial_kernel * time_kernel

        if eval_gradient:
            gradient_variance = K
            gradient_lengthscale = K * np.square(spatial_distance) / self.lengthscale
            n = time_distance / 2.0
            gradient_epsilon = -K * n * epsilon / (1.0 - epsilon)
            gradient = np.dstack([gradient_variance, gradient_lengthscale, gradient_epsilon])
            return K, gradient

        return K

    def diag(self, X):
        return np.full(X.shape[0], self.variance, dtype=np.float64)

    def is_stationary(self):
        return False

    @property
    def theta(self):
        return np.log([self.variance, self.lengthscale, self.epsilon])

    @theta.setter
    def theta(self, theta):
        self.variance = np.exp(theta[0])
        self.lengthscale = np.exp(theta[1])
        self.epsilon = np.exp(theta[2])

    @property
    def bounds(self):
        return np.log([
            list(self.variance_bounds),
            list(self.lengthscale_bounds),
            list(self.epsilon_bounds)
        ])


# ============================================================
# 7. PB2 NORMALIZATION
# ============================================================

def pb2_normalize(data, reference):
    data = np.asarray(data, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    minimum = np.min(reference, axis=0)
    maximum = np.max(reference, axis=0)
    denominator = maximum - minimum
    denominator = np.where(denominator < 1e-12, 1.0, denominator)

    return (data - minimum) / denominator


# ============================================================
# 8. PB2 STANDARDIZATION
# ============================================================

def pb2_standardize(y):
    y = np.asarray(y, dtype=np.float64)
    mean = np.mean(y)
    std = np.std(y)

    if std < 1e-12:
        return np.zeros_like(y)

    y = (y - mean) / std
    return np.clip(y, -2.0, 2.0)


# ============================================================
# 9. BUILD PB2 TRAINING DATA
# ============================================================

def build_pb2_training_data(history):
    if len(history) < 2:
        return None

    df = pd.DataFrame(history)
    if len(df) < 2:
        return None

    df = df.sort_values(["Trial", "Time"]).reset_index(drop=True)

    df["Reward_Delta"] = df.groupby("Trial")["Reward"].diff()
    df["Time_Delta"] = df.groupby("Trial")["Time"].diff()
    df = df[df["Time_Delta"] > 0].copy()

    if len(df) == 0:
        return None

    df["Reward_Before"] = df["Reward"] - df["Reward_Delta"]
    df["y"] = df["Reward_Delta"] / df["Time_Delta"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["y", "Reward_Before"])

    return df.reset_index(drop=True)


# ============================================================
# 10. PB2 HISTORY LENGTH SELECTION
# ============================================================

def pb2_select_history_length(X_raw, y_raw, bounds, num_fixed):
    n = X_raw.shape[0]
    MIN_LENGTH = 200

    if n < MIN_LENGTH:
        return n

    candidate_lengths = list(range(MIN_LENGTH, n + 1, 10))
    scores = []

    for length in candidate_lengths:
        X_part = X_raw[-length:]
        y_part = y_raw[-length:]

        fixed_part = X_part[:, :num_fixed]
        fixed_max = np.max(fixed_part, axis=0)
        fixed_min = np.min(fixed_part, axis=0)

        hp_min = np.array([bounds[name][0] for name in bounds], dtype=np.float64)
        hp_max = np.array([bounds[name][1] for name in bounds], dtype=np.float64)

        reference = np.vstack([
            np.concatenate([fixed_min, hp_min]),
            np.concatenate([fixed_max, hp_max])
        ])

        X = pb2_normalize(X_part, reference)
        y = pb2_standardize(y_part).reshape(-1, 1)

        kernel = TV_SquaredExp(variance=1.0, lengthscale=1.0, epsilon=0.1)
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-10, normalize_y=False, n_restarts_optimizer=0
        )

        try:
            model.fit(X, y)
            scores.append(model.log_marginal_likelihood_value_)
        except Exception:
            scores.append(-np.inf)

    if not scores:
        return min(n, MIN_LENGTH)

    best_idx = int(np.argmax(scores))
    return candidate_lengths[best_idx]


# ============================================================
# 11. PB2 UCB
# ============================================================

def pb2_ucb(mean_model, variance_model, x, fixed):
    c1 = 0.2
    c2 = 0.4

    n = max(1, mean_model.X_train_.shape[0])
    beta_t = c1 + max(0.0, np.log(c2 * n))
    kappa = np.sqrt(beta_t)

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    fixed = np.asarray(fixed, dtype=np.float64).reshape(-1)
    point = np.concatenate([fixed, x]).reshape(1, -1)

    try:
        mean = float(mean_model.predict(point)[0])
    except Exception:
        mean = -1e12

    try:
        _, std = variance_model.predict(point, return_std=True)
        std = float(std[0])
    except Exception:
        std = 0.0

    return mean + kappa * std


# ============================================================
# 12. PB2 ACQUISITION OPTIMIZATION
# ============================================================

def pb2_optimize_acquisition(mean_model, variance_model, fixed, num_hyperparameters, local_rng):
    NUM_RESTARTS = 10
    bounds = [(0.0, 1.0) for _ in range(num_hyperparameters)]
    best_value = -np.inf
    best_x = local_rng.uniform(0.0, 1.0, num_hyperparameters)

    def objective(x):
        value = pb2_ucb(mean_model, variance_model, x, fixed)
        return -value

    for _ in range(NUM_RESTARTS):
        x0 = local_rng.uniform(0.0, 1.0, num_hyperparameters)
        try:
            result = minimize(
                objective, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 200, "maxfun": 200, "disp": False}
            )
            x = np.clip(result.x, 0.0, 1.0)
            value = -result.fun

            if value > best_value:
                best_value = value
                best_x = x
        except Exception:
            continue

    return np.clip(best_x, 0.0, 1.0)


# ============================================================
# 13. FIT PB2 MODELS
# ============================================================

def pb2_fit_models(X_raw, y_raw, bounds, num_fixed):
    length = pb2_select_history_length(X_raw, y_raw, bounds, num_fixed)
    X_raw = X_raw[-length:]
    y_raw = y_raw[-length:]

    fixed_part = X_raw[:, :num_fixed]
    fixed_min = np.min(fixed_part, axis=0)
    fixed_max = np.max(fixed_part, axis=0)

    hp_min = np.array([bounds[name][0] for name in bounds], dtype=np.float64)
    hp_max = np.array([bounds[name][1] for name in bounds], dtype=np.float64)

    reference = np.vstack([
        np.concatenate([fixed_min, hp_min]),
        np.concatenate([fixed_max, hp_max])
    ])

    X = pb2_normalize(X_raw, reference)
    y = pb2_standardize(y_raw).reshape(-1, 1)

    kernel = TV_SquaredExp(variance=1.0, lengthscale=1.0, epsilon=0.1)
    model_mean = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-10, normalize_y=False, n_restarts_optimizer=0
    )

    try:
        model_mean.fit(X, y)
    except Exception as exc:
        print("PB2 GP fitting failed:", repr(exc))
        return None, None, None, None

    model_variance = model_mean
    return model_mean, model_variance, reference, length


# ============================================================
# 14. PB2 SELECT CONFIGURATION
# ============================================================

def pb2_select_config(history, current_configs, new_time, new_reward_before, bounds, local_rng):
    df = build_pb2_training_data(history)

    if df is None or len(df) < 2:
        return None

    hp_names = list(bounds.keys())
    X_raw = df[["Time", "Reward_Before"] + hp_names].values.astype(np.float64)
    y_raw = df["y"].values.astype(np.float64)
    num_fixed = 2

    if len(X_raw) > 1000:
        X_raw = X_raw[-1000:]
        y_raw = y_raw[-1000:]

    (model_mean, _, reference, history_length) = pb2_fit_models(X_raw, y_raw, bounds, num_fixed)

    if model_mean is None:
        return None

    X_norm = pb2_normalize(X_raw, reference)
    y_norm = pb2_standardize(y_raw).reshape(-1, 1)

    if current_configs is not None and len(current_configs) > 0:
        fake_rows = []
        for config in current_configs:
            row = np.concatenate([
                np.asarray([new_time, new_reward_before], dtype=np.float64),
                np.asarray([config[name] for name in hp_names], dtype=np.float64)
            ])
            fake_rows.append(row)

        fake_rows = np.asarray(fake_rows, dtype=np.float64)
        fake_rows_norm = pb2_normalize(fake_rows, reference)
        fake_y = np.zeros((len(fake_rows_norm), 1), dtype=np.float64)

        X_variance = np.vstack([X_norm, fake_rows_norm])
        y_variance = np.vstack([y_norm, fake_y])

        kernel = TV_SquaredExp(variance=1.0, lengthscale=1.0, epsilon=0.1)
        model_variance = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-10, normalize_y=False, optimizer="fmin_l_bfgs_b"
        )

        try:
            model_variance.fit(X_variance, y_variance)
        except Exception:
            model_variance = model_mean
    else:
        model_variance = model_mean

    new_fixed = np.asarray([new_time, new_reward_before], dtype=np.float64)
    new_fixed_norm = (new_fixed - reference[0, :num_fixed]) / (reference[1, :num_fixed] - reference[0, :num_fixed] + 1e-8)

    new_hp_norm = pb2_optimize_acquisition(
        mean_model=model_mean,
        variance_model=model_variance,
        fixed=new_fixed_norm,
        num_hyperparameters=len(hp_names),
        local_rng=local_rng
    )

    new_hp = {}
    for i, name in enumerate(hp_names):
        low, high = bounds[name]
        value = low + new_hp_norm[i] * (high - low)
        if name == "batch_size":
            value = int(np.clip(round(value), low, high))
        else:
            value = float(np.clip(value, low, high))
        new_hp[name] = value

    return new_hp


# ============================================================
# 15. INITIAL PB2 POPULATION
# ============================================================

def random_hparams(local_rng, local_py_rng):
    return {
        "gamma": float(local_rng.uniform(*PB2_BOUNDS["gamma"])),
        "learning_rate": float(local_rng.uniform(*PB2_BOUNDS["learning_rate"])),
        "batch_size": int(local_py_rng.randint(PB2_BOUNDS["batch_size"][0], PB2_BOUNDS["batch_size"][1])),
        "epsilon": float(local_rng.uniform(*PB2_BOUNDS["epsilon"])),
    }


# ============================================================
# 16. PB2 AGENT OBJECT
# ============================================================

@dataclass
class PB2Agent:
    agent_id: int
    trial_id: str
    hparams: dict
    agent_seed: int = None
    dqn: DQNAgent = None
    score: float = None

    def __post_init__(self):
        self.dqn = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hparams=self.hparams,
            agent_seed=self.agent_seed
        )


# ============================================================
# 17. BREAK PB2 TRIAL HISTORY
# ============================================================

def break_trial_history(history, old_trial_id, new_trial_id):
    for row in history:
        if row["Trial"] == old_trial_id:
            row["Trial"] = new_trial_id


# ============================================================
# 18. SAVE CSV
# ============================================================

def initialize_csv():
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def append_csv(iteration, generation, total_population_episodes, best_agent, wall_time, episodes_to_solve):
    with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            EXPERIMENT_ID,
            iteration,
            generation,
            total_population_episodes,
            best_agent.score,
            best_agent.hparams["learning_rate"],
            best_agent.hparams["batch_size"],
            best_agent.hparams["gamma"],
            best_agent.hparams["epsilon"],
            wall_time,
            episodes_to_solve
        ])


# ============================================================
# 19. MAIN PB2
# ============================================================

def runPB2(iteration_idx, base_seed):
    print("\n" + "=" * 70)
    print(f"PB2 - LunarLander-v3 | ITERATION {iteration_idx}")
    print("=" * 70)
    print(f"Population size      : {POPULATION_SIZE}")
    print(f"Episodes / generation: {EPISODES_PER_GEN}")
    print(f"Generations          : {GENERATIONS}")
    print(f"Total population eps : {GENERATIONS * POPULATION_SIZE * EPISODES_PER_GEN}")
    print("=" * 70)

    start_time = time.time()
    population = []
    
    iter_seed = base_seed + iteration_idx
    local_rng = np.random.default_rng(iter_seed)
    local_py_rng = random.Random(iter_seed)

    # ADDED: Print block to show the seed value of each agent at the beginning of the iteration
    print(f"\n--- Initializing Iteration {iteration_idx} Agents (Base Seed: {base_seed}, Iteration Seed: {iter_seed}) ---")
    for i in range(POPULATION_SIZE):
        agent_seed = iter_seed * 1000 + i
        print(f"  > Creating Agent {i} | Assigned Random Seed: {agent_seed}")
        
        hp = random_hparams(local_rng, local_py_rng)
        agent = PB2Agent(agent_id=i, trial_id=f"trial_{i}", hparams=hp, agent_seed=agent_seed)
        population.append(agent)

    history = []
    current_configs = []

    for gen in range(GENERATIONS):
        generation_number = gen + 1

        print("\n" + "-" * 70)
        print(f"GENERATION {generation_number}/{GENERATIONS}")
        print("-" * 70)

        agent_time = generation_number * EPISODES_PER_GEN

        for agent in population:
            print(f"  Agent {agent.agent_id} | Trial {agent.trial_id} | HP {agent.hparams}")
            agent.score = train_agent(agent.dqn)
            print(f"      Reward = {agent.score:.2f}")

            history.append({
                "Trial": agent.trial_id,
                "Time": float(agent_time),
                "gamma": float(agent.hparams["gamma"]),
                "learning_rate": float(agent.hparams["learning_rate"]),
                "batch_size": float(agent.hparams["batch_size"]),
                "epsilon": float(agent.hparams["epsilon"]),
                "Reward": float(agent.score)
            })

        population.sort(key=lambda a: a.score, reverse=True)
        best = population[0]

        total_population_episodes = generation_number * POPULATION_SIZE * EPISODES_PER_GEN
        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print(f"Generation       : {generation_number}")
        print(f"Best reward      : {best.score:.3f}")
        print(f"Best hyperparams : {best.hparams}")
        print(f"Population eps   : {total_population_episodes}")
        print(f"Wall time        : {elapsed / 60.0:.2f} min")
        print("=" * 70)

        is_solved = best.score >= STOP_THRESHOLD
        episodes_to_solve = total_population_episodes if is_solved else ""

        append_csv(
            iteration=iteration_idx,
            generation=generation_number,
            total_population_episodes=total_population_episodes,
            best_agent=best,
            wall_time=elapsed,
            episodes_to_solve=episodes_to_solve
        )

        if is_solved:
            print("\n" + "*" * 70)
            print("LUNARLANDER SOLVED")
            print(f"Reward = {best.score:.3f}")
            print(f"Generation = {generation_number}")
            print(f"Hyperparameters = {best.hparams}")
            print("*" * 70)
            break

        if generation_number >= GENERATIONS:
            break

        elites = population[:TOP_K]
        non_elites = population[TOP_K:]

        print("\nPB2 EXPLOIT / EXPLORE")
        print(f"Elites: {[(e.agent_id, round(e.score, 2)) for e in elites]}")

        current_configs = []

        for agent in non_elites:
            elite = random.choice(elites)
            old_trial = agent.trial_id
            new_trial = f"agent_{agent.agent_id}_trial_{generation_number}"

            print(f"\n  Agent {agent.agent_id}: clone elite {elite.agent_id}")

            agent.dqn.copy_weights_from(elite.dqn)
            break_trial_history(history, old_trial, new_trial)
            agent.trial_id = new_trial

            parent_reward = float(elite.score)
            parent_time = float(agent_time)

            history.append({
                "Trial": new_trial,
                "Time": parent_time,
                "gamma": float(elite.hparams["gamma"]),
                "learning_rate": float(elite.hparams["learning_rate"]),
                "batch_size": float(elite.hparams["batch_size"]),
                "epsilon": float(elite.hparams["epsilon"]),
                "Reward": parent_reward
            })

            if len(history) < 16:
                new_hp = copy.deepcopy(elite.hparams)
            else:
                new_hp = pb2_select_config(
                    history=history,
                    current_configs=current_configs,
                    new_time=parent_time,
                    new_reward_before=parent_reward,
                    bounds=PB2_BOUNDS,
                    local_rng=local_rng
                )
                if new_hp is None:
                    new_hp = copy.deepcopy(elite.hparams)

            agent.hparams = copy.deepcopy(new_hp)
            agent.dqn.update_hparams(new_hp)
            current_configs.append(copy.deepcopy(new_hp))

            print(f"      New HP = {new_hp}")

        gc.collect()

    elapsed = time.time() - start_time
    population.sort(key=lambda a: a.score, reverse=True)
    best = population[0]

    print("\n" + "=" * 70)
    print(f"PB2 FINISHED - ITERATION {iteration_idx}")
    print("=" * 70)
    print(f"Best reward: {best.score:.3f}")
    print(f"Best hyperparameters:")
    for key, value in best.hparams.items():
        print(f"    {key}: {value}")
    print(f"Runtime: {elapsed / 60.0:.2f} minutes")
    print(f"CSV: {CSV_FILENAME}")
    print("=" * 70)

    return best


# ============================================================
# 20. MAIN
# ============================================================

if __name__ == "__main__":
    
    base_seed = 42 
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            if config and "random_seed" in config:
                base_seed = int(config["random_seed"])
                print(f"Loaded base random seed {base_seed} from config.yaml")
    except Exception as e:
        print(f"Could not load config.yaml ({e}), defaulting to seed 42.")
        
    initialize_csv()

    for iteration_idx in range(1, NUM_ITERATIONS + 1):
        try:
            best_agent = runPB2(iteration_idx, base_seed)

        except KeyboardInterrupt:
            print(f"\nTraining interrupted during iteration {iteration_idx}.")
            break

        finally:
            tf.keras.backend.clear_session()
            gc.collect()
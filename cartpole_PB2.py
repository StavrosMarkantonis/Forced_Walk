# -*- coding: utf-8 -*-

"""
CartPole-v1
DQN + Population Based Bandits (PB2)
"""

# ============================================================
# 1. IMPORTS / THREAD CONFIGURATION
# ============================================================

import os

# THREADS = "4"

# os.environ["OMP_NUM_THREADS"] = THREADS
# os.environ["OPENBLAS_NUM_THREADS"] = THREADS
# os.environ["MKL_NUM_THREADS"] = THREADS
# os.environ["VECLIB_MAXIMUM_THREADS"] = THREADS
# os.environ["NUMEXPR_NUM_THREADS"] = THREADS

import csv
import copy
import gc
import random
import time
from collections import deque
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

ENV_NAME = "CartPole-v1"
STATE_DIM = 4
ACTION_DIM = 2

# Configuration Constants provided for CartPole
TOTAL_EPISODES = 0
NUM_ITERATIONS = 100
GENERATIONS_MAX = 2000
EPISODES_PER_GEN = 10
BUFFER_SIZE = 8_000
STOP_THRESHOLD = 500
EXPERIMENT_ID = "A01"

# Population settings
POPULATION_SIZE = 8
TOP_K = 3

# PB2 hyperparameter bounds
PB2_BOUNDS = {
    "gamma": (0.8, 0.999),
    "learning_rate": (0.0001, 0.01),
    "batch_size": (16, 1024),
    "epsilon": (0.01, 1.0),
}

CSV_FILENAME = "cartpole_PB2.csv"
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


# ============================================================
# 3. HIGH-PERFORMANCE OOP AGENT (Provided DQN)
# ============================================================

class DQNAgent:
    def __init__(self, state_dim, action_dim, hparams):
        self.hparams = hparams
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.score = None
        
        # Networks
        self.q_net = models.Sequential([
            layers.Input(shape=(state_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(action_dim)
        ])
        self.target_net = tf.keras.models.clone_model(self.q_net)
        self.target_net.set_weights(self.q_net.get_weights())
        
        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate=self.hparams["learning_rate"])
        self.loss_fn = losses.Huber()
        self.tau = 0.01

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
        """Compiled into a fast C++ TF Graph automatically."""
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        next_q_online = self.q_net(next_states, training=False)
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
        
        next_q_target = self.target_net(next_states, training=False)
        batch_indices = tf.range(tf.shape(next_actions)[0], dtype=tf.int32)
        next_action_indices = tf.stack([batch_indices, next_actions], axis=1)
        next_q = tf.gather_nd(next_q_target, next_action_indices)

        targets = rewards + gamma * next_q * (1.0 - dones)

        with tf.GradientTape() as tape:
            q_values = self.q_net(states, training=True)
            action_indices = tf.stack([batch_indices, actions], axis=1)
            q_action = tf.gather_nd(q_values, action_indices)
            loss = self.loss_fn(targets, q_action)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        
        # Soft Update
        for target_var, local_var in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
            target_var.assign(self.tau * local_var + (1.0 - self.tau) * target_var)

    @tf.function(
        reduce_retracing=True,
        input_signature=[tf.TensorSpec(shape=(1, STATE_DIM), dtype=tf.float32)]
    )
    def get_action(self, state):
        q_values = self.q_net(state, training=False)
        return tf.argmax(q_values[0])

    def update_hparams(self, new_hparams):
        """Instantly updates parameters without recompiling the model."""
        self.hparams = new_hparams
        tf.keras.backend.set_value(self.optimizer.learning_rate, self.hparams["learning_rate"])

    def copy_weights_from(self, elite_agent):
        """Exploits the elite model and safely flushes optimizer state."""
        self.q_net.set_weights(elite_agent.q_net.get_weights())
        self.target_net.set_weights(elite_agent.target_net.get_weights())
        # self.memory = copy.deepcopy(elite_agent.memory)
        
        # Instant Adam momentum flush (Zero-out 'm' and 'v' without recompiling)
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))


# ============================================================
# 4. MAIN PBT TRAINING FUNCTION
# ============================================================

def train_agent(agent):
    """
    Trains a single OOP agent for a set number of episodes.
    """
    env = gym.make("CartPole-v1")
    episodes = EPISODES_PER_GEN
    
    global TOTAL_EPISODES
    TOTAL_EPISODES += EPISODES_PER_GEN

    MIN_REPLAY_SIZE = 100
    batch_size = agent.hparams["batch_size"]
    epsilon = agent.hparams["epsilon"]
    gamma = agent.hparams["gamma"]

    # --- Training Loop ---
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_input = state[np.newaxis, :].astype(np.float32)
                action = agent.get_action(state_input).numpy()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.append((state, action, reward, next_state, terminated))
            state = next_state

            if len(agent.memory) >= MIN_REPLAY_SIZE:
                actual_batch_size = min(len(agent.memory), batch_size)
                batch = random.sample(agent.memory, actual_batch_size)
                states, actions, rewards, next_states, terms = map(np.array, zip(*batch))
                
                # Lightning fast compiled step
                agent.train_step(states, actions, rewards, next_states, terms, gamma)

    # --- Evaluation Phase ---
    eval_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            state_input = state[np.newaxis, :].astype(np.float32)
            action = agent.get_action(state_input).numpy()
            state, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            done = term or trunc
        eval_rewards.append(ep_reward)
    
    env.close()
    return np.mean(eval_rewards)


# ============================================================
# 5. PB2 TIME-VARYING SQUARED EXPONENTIAL KERNEL
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
# 6. PB2 NORMALIZATION / STANDARDIZATION
# ============================================================

def pb2_normalize(data, reference):
    data = np.asarray(data, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    minimum = np.min(reference, axis=0)
    maximum = np.max(reference, axis=0)
    denominator = maximum - minimum
    denominator = np.where(denominator < 1e-12, 1.0, denominator)

    return (data - minimum) / denominator


def pb2_standardize(y):
    y = np.asarray(y, dtype=np.float64)
    mean = np.mean(y)
    std = np.std(y)

    if std < 1e-12:
        return np.zeros_like(y)

    y = (y - mean) / std
    return np.clip(y, -2.0, 2.0)


# ============================================================
# 7. PB2 MODELING & OPTIMIZATION
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


def pb2_optimize_acquisition(mean_model, variance_model, fixed, num_hyperparameters):
    NUM_RESTARTS = 10
    bounds = [(0.0, 1.0) for _ in range(num_hyperparameters)]
    best_value = -np.inf
    best_x = np.random.uniform(0.0, 1.0, num_hyperparameters)

    def objective(x):
        value = pb2_ucb(mean_model, variance_model, x, fixed)
        return -value

    for _ in range(NUM_RESTARTS):
        x0 = np.random.uniform(0.0, 1.0, num_hyperparameters)
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


def pb2_select_config(history, current_configs, new_time, new_reward_before, bounds):
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
        num_hyperparameters=len(hp_names)
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
# 8. INITIAL PB2 POPULATION & AGENT DATACLASS
# ============================================================

def random_hparams():
    return {
        "gamma": float(np.random.uniform(*PB2_BOUNDS["gamma"])),
        "learning_rate": float(np.random.uniform(*PB2_BOUNDS["learning_rate"])),
        "batch_size": int(np.random.randint(PB2_BOUNDS["batch_size"][0], PB2_BOUNDS["batch_size"][1] + 1)),
        "epsilon": float(np.random.uniform(*PB2_BOUNDS["epsilon"])),
    }

@dataclass
class PB2Agent:
    agent_id: int
    trial_id: str
    hparams: dict
    dqn: DQNAgent = None
    score: float = None

    def __post_init__(self):
        self.dqn = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hparams=self.hparams
        )


# ============================================================
# 9. UTILITIES
# ============================================================

def break_trial_history(history, old_trial_id, new_trial_id):
    for row in history:
        if row["Trial"] == old_trial_id:
            row["Trial"] = new_trial_id

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
# 10. MAIN PB2
# ============================================================

def runPB2(iteration_idx):
    print("\n" + "=" * 70)
    print(f"PB2 - {ENV_NAME} | ITERATION {iteration_idx}")
    print("=" * 70)
    print(f"Population size      : {POPULATION_SIZE}")
    print(f"Episodes / generation: {EPISODES_PER_GEN}")
    print(f"Generations          : {GENERATIONS_MAX}")
    print(f"Total population eps : {GENERATIONS_MAX * POPULATION_SIZE * EPISODES_PER_GEN}")
    print("=" * 70)

    start_time = time.time()
    population = []

    for i in range(POPULATION_SIZE):
        hp = random_hparams()
        agent = PB2Agent(agent_id=i, trial_id=f"trial_{i}", hparams=hp)
        population.append(agent)

    history = []
    current_configs = []

    for gen in range(GENERATIONS_MAX):
        generation_number = gen + 1

        print("\n" + "-" * 70)
        print(f"GENERATION {generation_number}/{GENERATIONS_MAX}")
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
            print(f"{ENV_NAME.upper()} SOLVED")
            print(f"Reward = {best.score:.3f}")
            print(f"Generation = {generation_number}")
            print(f"Hyperparameters = {best.hparams}")
            print("*" * 70)
            break

        if generation_number >= GENERATIONS_MAX:
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
                    bounds=PB2_BOUNDS
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


if __name__ == "__main__":
    
    initialize_csv()

    for iteration_idx in range(1, NUM_ITERATIONS + 1):
        try:
            best_agent = runPB2(iteration_idx)

        except KeyboardInterrupt:
            print(f"\nTraining interrupted during iteration {iteration_idx}.")
            break

        finally:
            tf.keras.backend.clear_session()
            gc.collect()
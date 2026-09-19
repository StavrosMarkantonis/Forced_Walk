# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 13:04:17 2025

@author: smark
"""

import os
import yaml  # ADDED: to read from config.yaml

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

import gymnasium as gym
import numpy as np
import random
from collections import deque
from tensorflow.keras import layers, models, optimizers, losses
import csv
import copy
import gc
import time
import forced_walk as fw
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel 

# --- Configuration Constants ---
TOTAL_EPISODES = 0
NUM_ITERATIONS = 75
GENERATIONS_MAX = 2000
EPISODES_PER_GEN = 20
BUFFER_SIZE = 40_000
STOP_THRESHOLD = 200   # LunarLander is considered solved at 200
EXPERIMENT_ID = "A01"

CSV_FILENAME_PBT = 'lunarlander_PBT.csv'
CSV_FILENAME_FORCED = 'lunarlander_FW.csv'

# CSV Header Constant matching image_d15119.png exactly
CSV_HEADER = [
    'experiment_id', 'run_iteration', 'gen', 'total_episodes', 'score', 
    'lr', 'batch_size', 'gamma', 'epsilon', 'Max Iteration Episode', 'Iteration time'
]

# OOP AGENT
# --------------------------
class DQNAgent:
    # ADDED: agent_seed parameter for local layer initialization
    def __init__(self, state_dim, action_dim, hparams, agent_seed=None):
        self.hparams = hparams
        self.tau = 0.005  # ADDED: Polyak averaging blending factor
        
        # OPTIMIZATION 1: Pre-allocate NumPy arrays instead of using a Python deque
        self.memory_size = BUFFER_SIZE
        self.states = np.zeros((self.memory_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.memory_size, dtype=np.int32)
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        self.next_states = np.zeros((self.memory_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.memory_size, dtype=np.float32)
        self.mem_ptr = 0
        self.mem_count = 0
        
        self.score = None
        
        # ADDED: Local seed initializers to avoid tf.random.set_seed() global statements
        init1 = tf.keras.initializers.GlorotUniform(seed=agent_seed) if agent_seed else 'glorot_uniform'
        init2 = tf.keras.initializers.GlorotUniform(seed=agent_seed + 1 if agent_seed else None)
        init3 = tf.keras.initializers.GlorotUniform(seed=agent_seed + 2 if agent_seed else None)
        
        # Networks
        self.q_net = models.Sequential([
            layers.Input(shape=(state_dim,)),
            layers.Dense(128, activation="relu", kernel_initializer=init1),
            layers.Dense(128, activation="relu", kernel_initializer=init2),
            layers.Dense(action_dim, kernel_initializer=init3)
        ])
        self.target_net = tf.keras.models.clone_model(self.q_net)
        self.target_net.set_weights(self.q_net.get_weights())
        
        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate=self.hparams["learning_rate"])
        self.loss_fn = losses.Huber()

    def store_transition(self, state, action, reward, next_state, done):
        """Stores transitions directly into contiguous memory blocks."""
        idx = self.mem_ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        
        self.mem_ptr = (self.mem_ptr + 1) % self.memory_size
        self.mem_count = min(self.mem_count + 1, self.memory_size)

    def sample_memory(self, batch_size):
        """Samples vectorized arrays without zipping/unzipping tuples."""
        actual_size = min(self.mem_count, batch_size)
        indices = np.random.choice(self.mem_count, actual_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    @tf.function(reduce_retracing=True)
    def train_step(self, states, actions, rewards, next_states, dones, gamma):
        """Compiled into a fast C++ TF Graph automatically."""
        # OPTIMIZATION 2: Removed tf.cast overhead; inputs are already correct dtypes
        
        # Tell TF not to track gradient history for the target network
        with tf.name_scope('target_computation'):
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

        # ADDED: Soft target network updates (Polyak Averaging)
        for target_var, local_var in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
            target_var.assign(self.tau * local_var + (1.0 - self.tau) * target_var)

    @tf.function(reduce_retracing=True)
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
        
        # Deepcopying NumPy arrays is exponentially faster than deepcopying Deques of tuples
        self.states = np.copy(elite_agent.states)
        self.actions = np.copy(elite_agent.actions)
        self.rewards = np.copy(elite_agent.rewards)
        self.next_states = np.copy(elite_agent.next_states)
        self.dones = np.copy(elite_agent.dones)
        self.mem_ptr = elite_agent.mem_ptr
        self.mem_count = elite_agent.mem_count
        
        # Instant Adam momentum flush
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

# ---------------------------------------------------------
# 3. MAIN PBT TRAINING FUNCTION
# ---------------------------------------------------------
def train_and_eval(agent):
    env = gym.make("LunarLander-v3")
    episodes = EPISODES_PER_GEN
    
    global TOTAL_EPISODES
    TOTAL_EPISODES += EPISODES_PER_GEN
    print(f"Total Episodes: {TOTAL_EPISODES}")

    MIN_REPLAY_SIZE = 1000
    TRAIN_FREQ = 4 
    
    batch_size = agent.hparams["batch_size"]
    epsilon = agent.hparams["epsilon"]
    gamma = agent.hparams["gamma"]
    
    total_steps = 0

    # --- Training Loop ---
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            total_steps += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_input = state[np.newaxis, :].astype(np.float32)
                action = agent.get_action(state_input).numpy()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # OPTIMIZATION 3: Using the new O(1) NumPy storage
            agent.store_transition(state, action, reward, next_state, terminated)
            state = next_state

            if agent.mem_count >= MIN_REPLAY_SIZE and total_steps % TRAIN_FREQ == 0:
                # Fast NumPy slice extraction directly to inputs
                states, actions, rewards, next_states, terms = agent.sample_memory(batch_size)
                
                agent.train_step(states, actions, rewards, next_states, terms, gamma)
                
                # REMOVED: The manual target_net.set_weights() logic was deleted from here.

    # --- Evaluation Phase ---
    eval_episodes = 5 
    eval_rewards = []
    for _ in range(eval_episodes):
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

# -------------------------
# Forced Walk-PBT
# -------------------------
def get_fw_hparams(history_X, history_y, base_hparams, current_gen):
    """
    Uses the Forced Walk neural surrogate and proximal refinement to 
    intelligently mutate the hyperparameters.
    """
    parameters = [
        ("gamma", (0.8, 0.999), "float"),
        ("learning_rate", (0.0001, 0.01), "float"),
        ("batch_size", (16, 1024), "int"),
        ("epsilon", (0.01, 1.0), "float")
    ]
    
    # Instantiate study to access the Neural Network Surrogate and algorithms
    study = fw.create_fw_study(direction="maximize", terminate_value=STOP_THRESHOLD)
    
    # Populate the Experience Replay Buffer for the FW Surrogate
    for x, y in zip(history_X, history_y):
        # history_X format: [gen, gamma, lr, batch_size, epsilon]
        param_row = x[1:]
        study._append_training_data(param_row, y)
        
    # Train the FW Surrogate Neural Network
    study._train_value_network()
    
    base_init = [
        base_hparams["gamma"], 
        base_hparams["learning_rate"], 
        base_hparams["batch_size"], 
        base_hparams["epsilon"]
    ]
    
    # 1. Generate candidate mutations using Forced Walk's Bilevel Filtering
    scale = study.training_params.get("base_scale", 1.0)
    candidates = study._generate_parameters(base_init, current_gen, scale, parameters)
    
    # 2. Pick the absolute best candidate via the surrogate filter
    best_overall = study._filter_moves(candidates, 1, parameters)
    
    if best_overall:
        best_cand = best_overall[0]
    elif candidates:
        best_cand = candidates[0]
    else:
        return copy.deepcopy(base_hparams)
        
    return {
        "gamma": round(float(np.clip(best_cand[0], 0.8, 0.999)), 3),
        "learning_rate": round(float(np.clip(best_cand[1], 0.0001, 0.01)), 5),
        "batch_size": int(np.clip(best_cand[2], 16, 1024)),
        "epsilon": round(float(np.clip(best_cand[3], 0.01, 1.0)), 4)
    }

def run_forced_walk(current_iter, base_seed):
    global TOTAL_EPISODES
    TOTAL_EPISODES = 0
    POPULATION_SIZE = 8
    GENERATIONS = int(GENERATIONS_MAX / POPULATION_SIZE)
    TOP_K = 3  
    start_time = time.time()
    
    print("\n" + "="*40)
    print(f"STARTING FORCED WALK PBT: {POPULATION_SIZE} Agents, {GENERATIONS} Gens")
    print("="*40 + "\n")
    
    population = []
    iteration_logs = []
    history_X = []
    history_y = []
    
    if not os.path.isfile(CSV_FILENAME_FORCED):
        with open(CSV_FILENAME_FORCED, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADER)
    
    # ADDED: Calculate iteration specific seed to guarantee paired comparisons
    iter_seed = base_seed + current_iter
    local_rng = np.random.default_rng(iter_seed)
    local_py_rng = random.Random(iter_seed)
    
    # ADDED: Print block to show the seed value of each agent at the beginning of the iteration
    print(f"\n--- Initializing Iteration {current_iter} Agents (Base Seed: {base_seed}, Iteration Seed: {iter_seed}) ---")

    # Initialize OOP Population (LunarLander state/action dims)
    for i in range(POPULATION_SIZE):
        # ADDED: Unique seed per agent per iteration
        agent_seed = iter_seed * 1000 + i
        print(f"  > Creating Agent {i} | Assigned Random Seed: {agent_seed}")
        
        # CHANGED: Use isolated local generators for paired hparam initialization
        hparams = {
            "gamma": round(local_rng.uniform(0.8, 0.99), 3),
            "learning_rate": round(local_rng.uniform(0.0001, 0.01), 5),
            "batch_size": local_py_rng.randint(16, 1024),
            "epsilon": round(local_rng.uniform(0.01, 1), 4),
        }
        # CHANGED: Pass agent_seed directly down to the layer level
        population.append(DQNAgent(state_dim=8, action_dim=4, hparams=hparams, agent_seed=agent_seed))
    
    for gen in range(GENERATIONS):
        print(f"\n=== GENERATION {gen+1}/{GENERATIONS} ===")
        
        for agent in population:
            agent.score = train_and_eval(agent)
            
            history_X.append([
                gen + 1, 
                agent.hparams["gamma"], 
                agent.hparams["learning_rate"], 
                agent.hparams["batch_size"], 
                agent.hparams["epsilon"]
            ])
            history_y.append(agent.score)

        population.sort(key=lambda x: x.score, reverse=True)
        best = population[0]
        
        print(f"Best Reward: {best.score:.2f} | HP: {best.hparams}")

        iteration_logs.append([
            EXPERIMENT_ID, current_iter, gen + 1, TOTAL_EPISODES, best.score, 
            best.hparams['learning_rate'], best.hparams['batch_size'], 
            best.hparams['gamma'], best.hparams['epsilon'], 
            "", ""
        ])
              
        if best.score >= STOP_THRESHOLD:
            print(f"\n" + "*"*50)
            print(f"TARGET SCORE REACHED: {best.score:.2f} >= {STOP_THRESHOLD}")
            print(f"Stopping training early at Generation {gen+1}")
            print("*"*50 + "\n")
            break  
            
        elites = population[:TOP_K]
        rest = population[TOP_K:]
    
        for agent in rest:
            elite_1 = random.choice(elites)
            
            # CodeB's highly optimized NumPy deepcopy happens here seamlessly
            agent.copy_weights_from(elite_1)
            
            # 1. PBT Crossover logic
            if len(elites) > 1 and random.random() < 0.3:
                elite_2 = random.choice([e for e in elites if e is not elite_1])
                new_hparams = crossover(elite_1.hparams, elite_2.hparams)
            else:
                new_hparams = copy.deepcopy(elite_1.hparams)
            
            # 2. EXPLORE: Using Forced Walk neural surrogate mutation
            fw_mutated_hparams = get_fw_hparams(history_X, history_y, new_hparams, gen + 1)
            agent.update_hparams(fw_mutated_hparams) 
            
        gc.collect()

    iteration_time = int(time.time() - start_time)
    
    iteration_logs.append([
        EXPERIMENT_ID, current_iter, "", "", "", "", "", "", "", 
        TOTAL_EPISODES, iteration_time
    ])

    with open(CSV_FILENAME_FORCED, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(iteration_logs)

    print("\n" + "="*30)
    print("TRAINING FINISHED")
    print("="*30)

    tf.keras.backend.clear_session()
    gc.collect()


# -------------------------
# PBT Logic
# -------------------------
def crossover(parent_a, parent_b):
    child = {}
    for key in parent_a.keys():
        child[key] = copy.deepcopy(parent_a[key] if random.random() < 0.5 else parent_b[key])
    return child

def mutate(
    hparams,
    batch_bounds=(16, 1024),
    lr_bounds=(1e-4, 1e-2),
    gamma_bounds=(0.8, 0.999),
    epsilon_bounds=(0.01, 1)
):
    new = hparams.copy()
    
    if random.random() < 0.5:
        new["learning_rate"] = round(float(np.clip(new["learning_rate"] * random.choice([0.8, 1.2]), *lr_bounds)), 5)
    
    if random.random() < 0.5:
        new["batch_size"] = int(np.clip(new["batch_size"] * random.choice([0.8, 1.2]), *batch_bounds))
        
    if random.random() < 0.5:
        new["gamma"] = round(float(np.clip(new["gamma"] * random.choice([0.8, 1.2]), *gamma_bounds)), 3)

    if random.random() < 0.5:
        new["epsilon"] = round(float(np.clip(new["epsilon"] * random.choice([0.8, 1.2]), *epsilon_bounds)), 3)
    
    return new

def runPBT(current_iter, base_seed):
    global TOTAL_EPISODES
    TOTAL_EPISODES = 0
    POPULATION_SIZE = 8
    GENERATIONS = int(GENERATIONS_MAX/POPULATION_SIZE)
    TOP_K = 3  
    start_time = time.time()
    
    print("\n" + "="*40)
    print(f"STARTING PBT: {POPULATION_SIZE} Agents, {GENERATIONS} Gens")
    print("="*40 + "\n")
    
    iteration_logs = []
    
    if not os.path.isfile(CSV_FILENAME_PBT):
        with open(CSV_FILENAME_PBT, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADER)
    
    # ADDED: Calculate iteration specific seed to guarantee paired comparisons
    iter_seed = base_seed + current_iter
    local_rng = np.random.default_rng(iter_seed)
    local_py_rng = random.Random(iter_seed)

    # ADDED: Print block to show the seed value of each agent at the beginning of the iteration
    print(f"\n--- Initializing Iteration {current_iter} Agents (Base Seed: {base_seed}, Iteration Seed: {iter_seed}) ---")

    # Initialize OOP Population
    population = []
    for i in range(POPULATION_SIZE):
        # ADDED: Unique seed per agent per iteration
        agent_seed = iter_seed * 1000 + i
        print(f"  > Creating Agent {i} | Assigned Random Seed: {agent_seed}")
        
        # CHANGED: Use isolated local generators for paired hparam initialization
        hparams = {
            "gamma": round(local_rng.uniform(0.8, 0.99), 3),
            "learning_rate": round(local_rng.uniform(0.0001, 0.01), 5),
            "batch_size": local_py_rng.randint(16, 1024),
            "epsilon": round(local_rng.uniform(0.01, 1), 4),
        }
        # CHANGED: Pass agent_seed directly down to the layer level
        population.append(DQNAgent(state_dim=8, action_dim=4, hparams=hparams, agent_seed=agent_seed))
    
    for gen in range(GENERATIONS):
        print(f"\n=== GENERATION {gen+1}/{GENERATIONS} ===")
        
        for agent in population:
            agent.score = train_and_eval(agent)

        population.sort(key=lambda x: x.score, reverse=True)
        best = population[0]
        
        print(f"Best Reward: {best.score:.2f} | HP: {best.hparams}")

        iteration_logs.append([
            EXPERIMENT_ID, current_iter, gen + 1, TOTAL_EPISODES, best.score, 
            best.hparams['learning_rate'], best.hparams['batch_size'], 
            best.hparams['gamma'], best.hparams['epsilon'], 
            "", ""
        ])
             
        if best.score >= STOP_THRESHOLD:
            print(f"\n" + "*"*50)
            print(f"TARGET SCORE REACHED: {best.score:.2f} >= {STOP_THRESHOLD}")
            print(f"Stopping training early at Generation {gen+1}")
            print("*"*50 + "\n")
            break  
            
        elites = population[:TOP_K]
        rest = population[TOP_K:]
    
        for agent in rest:
            elite_1 = random.choice(elites)
            agent.copy_weights_from(elite_1)
            
            if len(elites) > 1 and random.random() < 0.3:
                elite_2 = random.choice([e for e in elites if e is not elite_1])
                new_hparams = crossover(elite_1.hparams, elite_2.hparams)
            else:
                new_hparams = copy.deepcopy(elite_1.hparams)
            
            agent.update_hparams(mutate(new_hparams)) 
            
        gc.collect()

    iteration_time = int(time.time() - start_time)
    
    iteration_logs.append([
        EXPERIMENT_ID, current_iter, "", "", "", "", "", "", "", 
        TOTAL_EPISODES, iteration_time
    ])

    with open(CSV_FILENAME_PBT, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(iteration_logs)

    print("\n" + "="*30)
    print("TRAINING FINISHED")
    print("="*30)

    tf.keras.backend.clear_session()
    gc.collect()    


# --------------------------
# Execution Block
# --------------------------
def run_and_log_iterations(algorithms, iterations):
    """
    Executes a list of dynamically passed algorithms multiple times.
    """
    global TOTAL_EPISODES
    
    # ADDED: Read base seed from YAML without altering global state
    base_seed = 42 # Fallback
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            if config and "random_seed" in config:
                base_seed = int(config["random_seed"])
                print(f"Loaded base random seed {base_seed} from config.yaml")
    except Exception as e:
        print(f"Could not load config.yaml ({e}), defaulting to seed 42.")
    
    for algo_func in algorithms:
        episode_counts = []
        
        print("\n" + "#"*50)
        print(f" INITIALIZING EXPERIMENT: {algo_func.__name__} ".center(50, "#"))
        print("#"*50)
        
        for i in range(iterations):
            current_iter = i + 1
            print(f"\n>>> Starting Iteration {current_iter} of {iterations} using {algo_func.__name__}...")        
            
            # CHANGED: Pass the base_seed directly to the algorithm
            algo_func(current_iter, base_seed)
            
            episode_counts.append(TOTAL_EPISODES)
            
            tf.keras.backend.clear_session()
            gc.collect()

        print("\n" + "="*40)
        print(f"ALL TOTAL_EPISODES RESULTS for {algo_func.__name__}:")
        print("="*40)
        print(episode_counts)
        print("="*40 + "\n")
    
if __name__ == "__main__":
    try:
        selected_algorithms = [
            run_forced_walk,
            # runPBT
        ]
        
        run_and_log_iterations(algorithms=selected_algorithms, iterations=NUM_ITERATIONS)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
        
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
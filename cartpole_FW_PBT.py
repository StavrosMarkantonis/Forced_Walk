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

# tf.config.threading.set_intra_op_parallelism_threads(7)

# --- Configuration Constants ---
TOTAL_EPISODES = 0
NUM_ITERATIONS = 100
GENERATIONS_MAX = 2000
EPISODES_PER_GEN = 10
BUFFER_SIZE = 8_000
STOP_THRESHOLD = 500
EXPERIMENT_ID = "A01"

CSV_FILENAME_PBT = 'cartpole_PBT.csv'
CSV_FILENAME_FORCED = 'cartpole_FW.csv'


CSV_HEADER = [
    'experiment_id', 'run_iteration', 'gen', 'total_episodes', 'score', 
    'lr', 'batch_size', 'gamma', 'epsilon', 'Max Iteration Episode', 'Iteration time'
]

# --------------------------
# HIGH-PERFORMANCE OOP AGENT
# --------------------------
class DQNAgent:
    # ADDED: agent_seed parameter for local layer initialization
    def __init__(self, state_dim, action_dim, hparams, agent_seed=None):
        self.hparams = hparams
        self.memory = deque(maxlen=BUFFER_SIZE)
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
        self.tau = 0.01

    @tf.function(reduce_retracing=True)
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
        self.memory = copy.deepcopy(elite_agent.memory)
        
        # Instant Adam momentum flush (Zero-out 'm' and 'v' without recompiling)
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

# ---------------------------------------------------------
# 3. MAIN PBT TRAINING FUNCTION
# ---------------------------------------------------------
def train_and_eval(agent):
    """
    Trains a single OOP agent for a set number of episodes.
    """
    env = gym.make("CartPole-v1")
    episodes = EPISODES_PER_GEN
    
    global TOTAL_EPISODES
    TOTAL_EPISODES += EPISODES_PER_GEN
    print(f"Total Episodes: {TOTAL_EPISODES}")

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
        # FW expects just the parameters without the 'gen' column
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


def run_forced_walk(current_iter, base_seed): # CHANGED: Accept base_seed
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
    best_per_gen = [] 
    iteration_logs = []
    history_X = []
    history_y = []
    
    if not os.path.isfile(CSV_FILENAME_FORCED):
        with open(CSV_FILENAME_FORCED, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADER)
    
    # ADDED: Calculate iteration specific seed to guarantee paired comparisons
    iter_seed = base_seed + current_iter
    local_rng = np.random.default_rng(iter_seed)
    local_py_rng = random.Random(iter_seed)
    
    # ADDED: Print block to show the seed value of each agent at the beginning of the iteration
    print(f"\n--- Initializing Iteration {current_iter} Agents (Base Seed: {base_seed}, Iteration Seed: {iter_seed}) ---")

    # Initialize OOP Population identically to PBT for fair evaluation
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
        population.append(DQNAgent(state_dim=4, action_dim=2, hparams=hparams, agent_seed=agent_seed))
    
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
            agent.copy_weights_from(elite_1)
            
            # 1. PBT Crossover logic
            if len(elites) > 1 and random.random() < 0.3:
                elite_2 = random.choice([e for e in elites if e is not elite_1])
                new_hparams = crossover(elite_1.hparams, elite_2.hparams)
            else:
                new_hparams = copy.deepcopy(elite_1.hparams)
            
            # 2. EXPLORE: Using Forced Walk instead of random PBT mutation
            fw_mutated_hparams = get_fw_hparams(history_X, history_y, new_hparams, gen + 1)
            agent.update_hparams(fw_mutated_hparams) 
            
        gc.collect()

    iteration_time = int(time.time() - start_time)
    
    iteration_logs.append([
        EXPERIMENT_ID, current_iter, "", "", "", "", "", "", "", 
        TOTAL_EPISODES, iteration_time
    ])

    with open(CSV_FILENAME_FORCED, 'a', newline='') as csvfile:
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

def runPBT(current_iter, base_seed): # CHANGED: Accept base_seed
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
        with open(CSV_FILENAME_PBT, 'w', newline='') as csvfile:
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
        population.append(DQNAgent(state_dim=4, action_dim=2, hparams=hparams, agent_seed=agent_seed))
    
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

    with open(CSV_FILENAME_PBT, 'a', newline='') as csvfile:
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
    (File saving is handled independently by the algorithm functions).
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
            
            # CHANGED: Execute the chosen algorithm dynamically and pass the base_seed
            algo_func(current_iter, base_seed)
            
            # Track the termination speed for console logging
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
        # Easily select which algorithms to run by commenting/uncommenting items in this list
        selected_algorithms = [
            run_forced_walk,
            # runPBT
        ]
        
        # Run the list of selected algorithms sequentially
        run_and_log_iterations(algorithms=selected_algorithms, iterations=NUM_ITERATIONS)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
        
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
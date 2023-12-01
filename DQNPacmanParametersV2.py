from dataclasses import dataclass
from typing import Any

@dataclass
class CharacterParams:
    space: int
    point: int
    wall: int
    enemy_safe: int
    pellet_v1: int
    pellet_v2: int
    enemy: int
    scared_enemy: int
    agent: int







@dataclass
class DeepQParams:
    gamma:  float
    batch_size: int
    buffer_size: int
    min_replay_size: int
    target_update_frequency: int
    optimizer_lr: float

@dataclass
class TrainingParams:
    epsilon_start: int
    epsilon_decay: int
    epsilon_end: int
    training_episodes: int

@dataclass 
class Sarsd:
    state:Any
    action:int
    reward:float
    next_state:Any
    done:int

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

NUM_ACTIONS = 4
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE =  10000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 80000
TARGET_UPDATE_FREQUENCY = 1000
OPTIMIZER_LR = 5e-4
EPISODES = 3000
RUN_DURATION = 5 #Hours 


# 28 Across 31 Tall 1: Empty Space 2: Tic-Tak 3: Wall 4: Ghost safe-space 5: Special Tic-Tak

PATH = 1
GOAL = 2
WALL = 3
GHOST_SAFE_SPACE = 4
SPECIAL_PELLETS_V1 = 5
SPECIAL_PELLETS_V2 = 6
ENEMY = 7
SCARED_GHOSTS = 8
AGENT = 9

DQN_ALG = 1
DDQN_ALG = 2
DQN_ALG_PR = 3

CONST_EP =1
LINEAR_EP = 2
CURVE_EP  = 3
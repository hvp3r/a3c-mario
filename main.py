
import os
import torch.multiprocessing as mp

from collections import deque
from environment import Environment
from agent import GlobalAgent, LocalAgent, TestAgent

ALPHA = 1e-4

LOCAL_STEPS = 50
GLOBAL_STEPS = 500000
MAX_ACTIONS = 200

GAMMA = 0.9
TAU = 1.0
BETA = 0.01

SAVE_EACH = 5
DIR_PATH = os.path.realpath(os.path.dirname(__file__))
SAVE_PATH = DIR_PATH + "/session/checkpoint"


def run_local_agent(index, global_model, optimizer):
    save = index == 0

    local_agent = LocalAgent(index, global_model, optimizer, LOCAL_STEPS, GLOBAL_STEPS, 
        GAMMA, TAU, BETA, SAVE_EACH, SAVE_PATH, save)

    print("Process {} running... ".format(index))
    local_agent.train()


def run_test_agent(num_inputs, num_actions):
    agent = TestAgent(num_inputs, num_actions, SAVE_PATH)
    actions = deque(maxlen = MAX_ACTIONS)
    env = Environment()
    print("Test Process running...")

    while True:
        agent.load()
        step = 0
        actions.clear()   
        done = False
        state = env.reset()    
        while not done:
            step += 1
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            actions.append(action)
            
            env.render()
            if step > GLOBAL_STEPS or actions.count(actions[0]) == actions.maxlen:
                done = True


if __name__ == '__main__':
    if not os.path.exist("session"):
        os.makedirs("session")

        
    mp.set_start_method("spawn")

    env = Environment()
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    global_agent = GlobalAgent(num_inputs, num_actions, ALPHA, SAVE_PATH)

    model = global_agent.model
    model.share_memory()

    optimizer = global_agent.optimizer
    
    num_processes = os.cpu_count()
    processes = []

    for rank in range(num_processes):
        p = mp.Process(target=run_local_agent, args=(rank, model, optimizer))
        p.start()
        processes.append(p)
    
    p = mp.Process(target=run_test_agent, args=(num_inputs, num_actions))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

import gym_super_mario_bros


from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import Reward, SkipFrame


ENV_NAME = "SuperMarioBros-1-1-v0"

def Environment():
    env = gym_super_mario_bros.make(ENV_NAME)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = Reward(env)
    env = SkipFrame(env)
    return env


from environment import Environment
from agent import TestAgent


env = Environment()

location = "session/trained"
agent = TestAgent(env.observation_space.shape[0], env.action_space.n, location)

state = env.reset()
done = False
while not done:
    action= agent.choose_action(state)
    state, reward, done, info = env.step(action)
    env.render()

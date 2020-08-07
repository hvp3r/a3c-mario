
import os
import torch as T
import torch.nn.functional as F

from environment import Environment

from utils.model.network import GenericModel
from utils.optim.Optimizer import GlobalAdam

class GlobalAgent(object):
    def __init__(self, num_inputs, num_actions, alpha, save_path):
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
        self.model = GenericModel(num_inputs, num_actions).to(self.device)
        self.optimizer = GlobalAdam(self.model.parameters(), alpha)
        self.save_path = save_path
        self.__load()

    def __load(self):
        if os.path.isfile(self.save_path):
            self.model.load_state_dict(T.load(self.save_path, map_location=self.device))


class LocalAgent(object):
    def __init__(self, index, global_model, optimizer, local_steps, global_steps, gamma,
        tau, beta, save_each, save_path, save):
        self.index = index
        self.env = Environment()
        self.num_inputs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        
        self.global_model = global_model  
        
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
        self.model = GenericModel(self.num_inputs, self.num_actions).to(self.device)
        self.optimizer = optimizer

        self.local_steps = local_steps
        self.global_steps = global_steps

        self.gamma = gamma
        self.tau = tau
        self.beta = beta

        self.save_each = save_each
        self.save_path = save_path
        self.save = save
    
        self.done = True

    def train(self):
        episode = 1
        step = 0
        while True:
            state = self.env.reset()

            print("Process {}; Episode {}".format(self.index, episode))
            if episode % self.save_each == 0: self.__save()

            self.__update()
            self.__reset()
            state, step, R = self.__play(state, step)

            self.optimizer.zero_grad()
            
            loss = self.__loss(R)
            loss.backward()

            params = zip(self.model.parameters(), self.global_model.parameters())
            for local_param, global_param in params:
                if global_param.grad is not None: break
                global_param._grad = local_param.grad

            self.optimizer.step()
            episode += 1
    

    def __reset(self):
        if self.done: self.model.clear()

        self.values = []
        self.policies = []
        self.rewards = []
        self.entropies = []

    def __save(self):
        if self.save:
            T.save(self.global_model.state_dict(), self.save_path)

    def __update(self):
        self.model.load_state_dict(self.global_model.state_dict())

    def __choose_action(self, observation):
        observation = T.from_numpy(observation).to(self.device)
        probs, value = self.model(observation)
        probs = F.softmax(probs, dim=1) 
        actions_probs = T.distributions.Categorical(probs)
        action = actions_probs.sample().item()

        return action, probs, value

    def __play(self, state, current_step):
        for _ in range(self.local_steps):
            current_step += 1
            action, probs, value = self.__choose_action(state)
            action_log = F.log_softmax(probs, dim=1)
            entropy = -(action * action_log).sum(1, keepdim=True)

            state, reward, self.done, _ = self.env.step(action)
            
            self.values.append(value)
            self.policies.append(action_log[0, action])
            self.rewards.append(reward)
            self.entropies.append(entropy)

            self.done = current_step > self.global_steps

            if self.done:
                current_step = 0
                state = self.env.reset()
                break
        
        R = T.zeros((1, 1), dtype=T.float).to(self.device)
        if not self.done: 
            _, R = self.model(T.from_numpy(state).to(self.device))
        return state, current_step, R

    def __loss(self, R):
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        gae = T.zeros((1, 1), dtype=T.float).to(self.device)
        
        for value, log_policy, reward, entropy in list(zip(self.values, self.policies, self.rewards, self.entropies))[::-1]:
            gae = gae * self.gamma * self.tau
            gae = gae + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * self.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        return -actor_loss + critic_loss - self.beta * entropy_loss
        

class TestAgent(object):
    def __init__(self, num_inputs, num_actions, save_path):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
        self.save_path = save_path
        
        self.model = GenericModel(self.num_inputs, self.num_actions)
        self.model.to(self.device)
        self.model.eval()

        self.load()

    def choose_action(self, observation):
        observation = T.from_numpy(observation).to(self.device)
        probs, value, = self.model(observation)
        actions = F.softmax(probs, dim=1)
        action = T.argmax(actions).item()
        return action

    def load(self):
        if os.path.isfile(self.save_path):
            self.model.load_state_dict(T.load(self.save_path,  map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

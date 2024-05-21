import copy
import random
import time
import torch
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from plot import LivePlot


class ReplayMemory:

    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0

    # replay memory can get very, very large in training
    # store everything to cpu replay memory
    # then we push it back to gpu
    def insert(self, transition):  # tuple to contain state, action, reward, next_state
        transition = [item for item in transition]  # list comprehension
        #print(transition)
        #transition = torch.FloatTensor(transition)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)

    def sample(self, batch_size=32):  # how big a sample of data you want to sample
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)  # random sample of transitions from memory of size batch size
        batch = zip(*batch)  # taking the batches zipping together

        return [torch.cat(items).to(self.device) for items in batch]  # pushing to gpu

    def can_sample(self, batch_size=32):  # ask the memory object if it has enough to sample batch size
        return len(self.memory) >= batch_size * 10

    def __len__(self):  # overwriting the len will give only len of memory not the len of the class
        return len(self.memory)


class Agent:

    # epsilon is for randomness - epsilon greedy approach
    # min epsilon the lowest value of epsilon can have by decaying - exploration
    # nb_warmup - period over which epsilon will decay
    def __init__(self, model, device, epsilon=1.0, min_epsilon=0.1, nb_warmup=10000, nb_actions=None,
                 memory_capacity=10000, batch_size=32, learning_rate=0.00025):
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()  # we want model we can evaluate
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (
                ((epsilon - min_epsilon) / nb_warmup) * 2)  # pretty close to nb_warmup steps, can be any formula
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99  # discount future rewards
        self.nb_actions = nb_actions

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print("start epsilon is", self.epsilon)
        print("Epsilon decay", self.epsilon_decay)

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:  # epsilon greedy approach
            return torch.randint(self.nb_actions,(1, 1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True)  # returns max index - which is the action

    def train(self, env, epochs):
        stats = {'Returns': [], 'AvgReturns': [], 'EpsilonCheckpoint': []}

        plotter = LivePlot()

        for epoch in range(1, epochs + 1):
            state = env.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.get_action(state)

                next_state, reward, done, info = env.step(action)

                self.memory.insert([state, action, reward, done, next_state])

                if self.memory.can_sample(self.batch_size):
                    # bring the batch not single rows of data, here zip comes into play
                    # '_b' because they are batches
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    print(state_b.type(), action_b.type(), reward_b.type(), done_b.type(), next_state_b.type())
                    qsa_b = self.model(state_b).gather(1, action_b)  # read it
                    #print(next_state_b.shape, next_state_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=1, keepdim=True)[0]
                    target_b = reward_b + (~done_b).float() * self.gamma * next_qsa_b  # if done is True there is no next state,so we are negating it
                    loss = F.mse_loss(qsa_b, target_b)  # loss function
                    self.model.zero_grad()
                    loss.backward()  # back propagation
                    self.optimizer.step()

                state = next_state  # state update
                ep_return += reward.item()

            stats['Returns'].append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            if epoch % 10 == 0:
                self.model.save_the_model()
                print("  ")

                average_returns = np.mean(stats['Returns'][-100:])

                stats['AvgReturns'].append(averge_returns)
                stats['EpsilonCheckpoint'].append(self.epsilon)

                if len(stats['Returns']) > 100:
                    print("Episode {}: Average Return: {} Epsilon{}".format(epoch, np.mean(stats['AvgReturns'][-100:]), self.epsilon))
                else:
                    print("Episode {}: Average Return: {} Epsilon{}".format(epoch, np.mean(stats['AvgReturns'][-1:]), self.epsilon))

            if epoch % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict()) # target model updated with our model
                plotter.update_polt(stats)

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")

        return stats

    def test(self, env):
        for epoch in range(1,3):
            state = env.reset()
            done = False

            for _ in range(1000):
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break


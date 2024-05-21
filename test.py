import os
import gym
import numpy as np
from PIL import Image
import torch

from breakout import *
from model import AtariNet
from agent import Agent

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

environment = DQNBreakout(device=device, render_mode='human')

model = AtariNet(nb_actions=4)
model.to(device)
model.load_the_model()

agent = Agent(model=model, device=device, epsilon=1.0, nb_warmup=5000, nb_actions=4, learning_rate=0.00001,
              memory_capacity=1000000, batch_size=64)

agent.test(env=environment)



# state = environment.reset()
# # state = np.array(state)
# # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
# # print(state.shape)
# action_probs = model.forward(state).detach()
# print(action_probs, torch.argmax(action_probs, dim=1, keepdim=True))
#
# #state = environment.reset()
#
# # for _ in range(100):
# #     action = environment.action_space.sample()
# #     state, reward, done, info = environment.step(action)

import collections
import cv2
import gym
import numpy as np
from PIL import Image
import torch


class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
        env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)

        super(DQNBreakout, self).__init__(env)

        self.image_shape = (84, 84)  # standard
        self.repeat = repeat
        self.lives = env.ale.lives()
        self.frame_buffer = []

    def step(self, action):
        total_reward = 0
        done = False

        for _ in range(self.repeat):  # _ is the throw away variable
            obs, rew, done, trunc, info = self.env.step(action)

            total_reward += rew

            current_lives = info['lives']

            if current_lives < self.lives:
                total_reward = total_reward - 1
                self.lives = current_lives

            self.frame_buffer.append(obs)

            if done:
                break
        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_observation(max_frame)
        #max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward).view(-1, 1).float()
        #total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(-1, 1)
        #done = done.to(self.device)

        return max_frame, total_reward, done, info

    def reset_env(self):
        self.frame_buffer = []

        observation, _ = self.env.reset()  # _ is used when you want to throw away the variable only observation is needed

        self.lives = self.env.ale.lives()
        observation = self.process_observation(observation)

        return observation

    def process_observation(self, observation):
        #Add content
        img = Image.fromarray(observation)  # array of num from Image pulled from PIL
        img = img.resize(self.image_shape)  # shrink the image to 84x84
        img = img.convert("L")  # grayscale
        img = np.array(img)  # into array
        img = torch.from_numpy(img)  # tensor
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)  # adds two dimensions
        img = img / 255.0  # normalization to value between 0 and 1

        #img = img.to(self.device)
        return img

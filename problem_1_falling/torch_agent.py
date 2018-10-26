import torch as th
import numpy as np
from numpy.random import choice
from utils import read_cfg
from falling_objects_env import FallingObjects, PLAYER_KEYS, ACTIONS
import cv2
import json
import os
from ast import literal_eval
from cnnlstm import SqueezenetLSTM
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from collections import deque

def train(model, x, y, frame_size, learning_rate):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    # iterate through all dataset
    x = Variable(torch.from_numpy(x).float())
    y = Variable(torch.from_numpy(y).float())
    optimizer.zero_grad()
    logit = model(x, frame_size)
    loss = loss_function(logit[0], y)
    loss.backward()
    optimizer.step()
    #print("Train MSE: {}".format(loss))
    #val_loss = validate(model, x_train, batch_size, frame_size, valid_ids)
    #print("Epoch: {}; Valid MSE: {}".format(i, np.average(losses)))
    #torch.save(model, 'model.pt')

def predict(model, x, frame_size):
    y_pred = []
    x = Variable(torch.from_numpy(x).float())
    model.eval()
    logit = model(x, frame_size)
    return logit.detach().numpy()[0]

class TorchAgent:
    def __init__(self, max_action: int, learning_rate=0.1, discount=0.9, config_file='configs/default.yaml', frame_size=8,
        epsilon=0.3, epsilon_min=0.02, epochs=30000, moves_per_epoch=1000, batch_size=8, net_lr=0.001):
        self.max_action = max_action
        self.actions = [1, 2, 3] #[x for x in range(self.max_action)]
        self.learning_rate = learning_rate
        self.discount = discount
        self.config_file = config_file
        self.frame_size = frame_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epochs = epochs
        self.moves_per_epoch = moves_per_epoch
        try:
            self.model = torch.load('configs/model.pt')
            print("Loaded model")
        except:
            self.model = SqueezenetLSTM()
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size
        self.net_lr = net_lr
        self.done_pre = False

    def replay(self):
        indices = np.random.choice(len(self.memory), size=min(self.batch_size, len(self.memory)))
        batch = [self.memory[i] for i in indices]
        for state, action, r, statep in batch:
            target = (r + self.discount * np.amax(predict(self.model, statep, self.frame_size)))
            prediction = predict(self.model, state, self.frame_size)
            prediction[action] = target
            states.append(state)
            predictions.append(prediction)
            train(self.model, state, prediction, self.frame_size, self.net_lr)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * 0.9999

    def epsilon_greedy(self, state, legal_actions, epsilon):
        chances = int(100 * epsilon) * [0] + int(100 * (1 - epsilon)) * [1]
        r = choice(chances)
        if r == 0:
            return choice([0, 1, 2])
        else:
            res = predict(self.model, state, self.frame_size) 
            return np.argmax(res)   

    def qlearning(self):
        train_scores = []
        eval_scores = []

        cfg = read_cfg(self.config_file)

        all_scores = []

        for train_ep in range(self.epochs):
            if train_ep <= 10:
                self.epsilon = 1
            elif self.done_pre == False:
                self.done_pre = True
                self.epsilon = 0.25
            score = 0
            env = FallingObjects(cfg)
            obs = env.reset()
            obs, _ = self.get_state(obs)
            stack_frame = deque([obs for _ in range(self.frame_size)], maxlen=self.frame_size)
            state, stack_frame = self.get_frame(stack_frame, obs)
            state = np.reshape(state, [1, 1, self.frame_size, 86, 86])

            for i in range(self.moves_per_epoch):
                actions = self.actions
                action = self.epsilon_greedy(state, actions, self.epsilon)

                obs, r, done, _ = env.step(actions[action])
                obs, r = self.get_state(obs)
                print("Move: {}; action: {}; reward: {}; epsilon: {}".format(i, actions[action], r, self.epsilon))

                statep, stack_frame = self.get_frame(stack_frame, obs)
                statep = np.reshape(statep, [1, 1, self.frame_size, 86, 86])
                score += r

                self.memory.append((state, action, r, statep))
                
                state = statep

                if train_ep > 10:
                    self.replay()

            print("Episode: {}; score: {}".format(train_ep, score))
            all_scores.append(score)
            if train_ep % 20 == 0 and train_ep > 0:
                print("Mean score for the last 200 epochs: {}".format(np.average(all_scores[:-200])))
                torch.save(self.model, 'model.pt')

    def get_state(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        gray[gray > 0] = 255
        s = np.where(np.logical_and(observation[:, :, 0] > 0, observation[:, :, 1] == 0, observation[:, :, 2] == 0))
        rows = s[0]
        cols = s[1]

        o = np.where(np.logical_and(observation[:, :, 0] > 0, observation[:, :, 1] > 0, observation[:, :, 2] > 0))
        o_rows = o[0]
        o_cols = o[1]

        min_rows = np.min(rows)
        min_cols = np.min(cols)
        max_rows = np.max(rows)
        max_cols = np.max(cols)
        if o_rows != [] and o_cols != []:
            min_o_rows = np.min(o_rows)
            min_o_cols = np.min(o_cols)
            max_o_rows = np.max(o_rows)
            max_o_cols = np.max(o_cols)
            if min_cols >= min_o_cols and max_cols <= max_o_cols:
                r = -100
            elif min_cols <= min_o_cols and max_cols > min_o_cols:
                r = -50
            elif min_cols <= max_o_cols and max_cols > max_o_cols:
                r = -50
            else:
                r = 0#observation.shape[1] - abs(observation.shape[1] // 2 - (min_cols + (max_cols - min_cols) // 2))
            return gray, r
        else:
            return gray, 0

    def get_frame(self, stack_frame, state):
        stack_frame.append(state)
        return np.stack(stack_frame, axis=2), stack_frame

    def act(self, observation: np.ndarray):
        """
        :param observation: numpy array of shape (width, height, 3) *defined in config file
        :return: int between 0 and max_action
        """
        state, _ = self.get_state(observation)
        actions = self.actions
        action = self.epsilon_greedy(state, actions, 0)
        return action


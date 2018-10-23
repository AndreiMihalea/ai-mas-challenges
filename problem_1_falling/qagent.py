import numpy as np
from numpy.random import choice
from utils import read_cfg
from falling_objects_env import FallingObjects, PLAYER_KEYS, ACTIONS
import cv2
import json
import os
from ast import literal_eval

class QAgent:
    def __init__(self, max_action: int, learning_rate=0.1, discount=0.9, config_file='configs/default.yaml',
        epsilon=0.7, epsilon_min=0.02, epochs=30000, moves_per_epoch=1000, q_path='configs/q.txt', load=True):
        self.max_action = max_action
        self.actions = [1, 2, 3]
        self.learning_rate = learning_rate
        self.discount = discount
        self.config_file = config_file
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epochs = epochs
        self.moves_per_epoch = moves_per_epoch
        self.q_path = q_path
        self.load = load
        if self.load:
            self.load_q()
        else:
            self.Q = {}

    def get_state(self, observation):
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
                r = -10 - max_o_rows * 0.1
            elif min_cols <= min_o_cols and max_cols > min_o_cols:
                r = -5 - max_o_rows * 0.1
            elif min_cols <= max_o_cols and max_cols > max_o_cols:
                r = -5 - max_o_rows * 0.1
            else:
                r = 5#observation.shape[1] - abs(observation.shape[1] // 2 - (min_cols + (max_cols - min_cols) // 2))
            return (min_cols, max_cols, min_o_cols, max_o_cols, max_o_rows), r
        else:
            return (min_cols, max_cols, 0, 0, 87), 0

    def epsilon_greedy(self, Q, state, legal_actions, epsilon):
        chances = int(100 * epsilon) * [0] + int(100 * (1 - epsilon)) * [1]
        r = choice(chances)
        if r == 0:
            return choice(legal_actions)
        else:
            maximum = -float('inf')
            argmax = legal_actions[0]
            for action in legal_actions:
                if Q.get((state, action), 0) > maximum:
                    maximum = Q.get((state, action), 0)
                    argmax = action
            return argmax

    def qlearning(self):
        train_scores = []
        eval_scores = []

        cfg = read_cfg(self.config_file)

        all_scores = []

        for train_ep in range(self.epochs):
            score = 0
            env = FallingObjects(cfg)
            obs = env.reset()
            state, _ = self.get_state(obs)

            for i in range(self.moves_per_epoch):
                actions = self.actions
                action = self.epsilon_greedy(self.Q, state, actions, self.epsilon)

                obs, r, done, _ = env.step(action)
                statep, r = self.get_state(obs)
                if train_ep > 1:
                    print(statep, r)
                    cv2.imshow('hehe', obs)
                    cv2.waitKey(0)
                score += r

                maximum = -float('inf')
                actionsp = self.actions
                for actionp in actionsp:
                    if self.Q.get((statep, actionp), 0) > maximum:
                        maximum = self.Q.get((statep, actionp), 0)   

                self.Q[(state, action)] = self.Q.get((state, action), 0) + self.learning_rate * (r + self.discount * maximum - self.Q.get((state, action), 0))

                state = statep

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= 0.99999

            print("Epoch: {}; Score: {}; Epsilon: {}".format(train_ep, score, self.epsilon))
            all_scores.append(score)
            if train_ep % 200 == 0 and train_ep > 0:
                self.save_q()
                print("Mean score for the last 200 epochs: {}".format(np.average(all_scores[:-200])))
    def preprocess(self, obs):
        red = obs[:, :, 2]
        green = obs[:, :, 1]
        blue = obs[:, :, 0]
        res = np.zeros((obs.shape[0], obs.shape[1]))
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if green[i, j] < 100 and red[i, j] < 100 and blue[i, j] > 80:
                    res[i, j] = 1
                elif green[i, j] > 40 and red[i, j] > 40 and blue[i, j] > 40:
                    res[i, j] = 1
        return res

    def save_q(self):
        with open(self.q_path, 'w') as qfile:
            for k in self.Q:
                qfile.write(str(k) + "; " + str(self.Q[k]) + "\n")

    def load_q(self):
        self.Q = {}
        try:
            with open(self.q_path, 'r') as qfile:
                lines = qfile.readlines()
                for l in lines:
                    k = l.split(';')[0]
                    k = literal_eval(k)
                    v = l.split(';')[1]
                    v = float(v)
                    self.Q[k] = v
        except:
            print("Reading file failed. Initializing Q with {}")

    def act(self, observation: np.ndarray):
        """
        :param observation: numpy array of shape (width, height, 3) *defined in config file
        :return: int between 0 and max_action
        """
        state, _ = self.get_state(observation)
        actions = self.actions
        action = self.epsilon_greedy(self.Q, state, actions, self.epsilon)
        return action

import numpy as np
from numpy.random import choice
from utils import read_cfg
from falling_objects_env import FallingObjects, PLAYER_KEYS, ACTIONS
import cv2
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.layers.convolutional_recurrent import ConvLSTM2D

class DemoAgent:
    def __init__(self, max_action: int, learning_rate=0.1, discount=0.99, config_file='configs/default.yaml', frames=5,
        epsilon=0.9, epsilon_min=0.02, epochs = 30000, moves_per_epoch=1000, net_learning_rate=0.01, batch_size=16):
        self.max_action = max_action
        self.actions = [1, 2, 3] #[x for x in range(self.max_action)]
        self.learning_rate = learning_rate
        self.discount = discount
        self.config_file = config_file
        self.frames=frames
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epochs = epochs
        self.moves_per_epoch = moves_per_epoch
        self.net_learning_rate = net_learning_rate
        try:
            self.model = load_model('configs/model.h5')
        except:
            self.model = self.create_model()
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size
        self.done_pre = False


    def create_model(self):
        model = Sequential()
        '''
        # This is too slow on my poor laptop
        model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                           input_shape=(self.frames, 86, 86, 1),
                           padding='same', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                           input_shape=(self.frames, 86, 86, 1),
                           padding='same', return_sequences=True))
        model.add(BatchNormalization())
        model.add(Flatten())
        '''
        model.add(Dense(32, input_dim=5, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.actions), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.net_learning_rate))
        model.summary()
        return model

    def get_frame(self, stack_frame, state):
        stack_frame.append(state)
        return np.stack(stack_frame, axis=2), stack_frame

    def replay(self):
        indices = np.random.choice(len(self.memory), size=min(self.batch_size, len(self.memory)))
        batch = [self.memory[i] for i in indices]
        states = []
        predictions = []
        for state, action, r, statep in batch:
            target = (r + self.discount * np.amax(self.model.predict(np.array([statep]))[0]))
            predict = self.model.predict(np.array([state]))
            predict[0][action] = target
            states.append(state)
            predictions.append(predict[0])
        states = np.array(states)
        predictions = np.array(predictions)
        self.model.fit(states, predictions, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * 0.9999

    def epsilon_greedy(self, state, legal_actions, epsilon):
        chances = int(100 * epsilon) * [0] + int(100 * (1 - epsilon)) * [1]
        r = choice(chances)
        if r == 0:
            return choice(len(legal_actions))
        else:
            res = self.model.predict(np.array([state]))
            return np.argmax(res[0])

    def qlearning(self):
        cfg = read_cfg(self.config_file)
        all_scores = []
        for train_ep in range(self.epochs):
            if train_ep <= 10:
                self.epsilon = 0.02
            elif self.done_pre == False:
                self.done_pre = True
                self.epsilon = 0.6
            score = 0
            env = FallingObjects(cfg)
            obs = env.reset()
            obs, _ = self.get_state(obs)
            #stack_frame = deque([obs for _ in range(self.frames)], maxlen=self.frames)
            #state, stack_frame = self.get_frame(stack_frame, obs)
            #state = np.reshape(state, [1, self.frames, 86, 86, 1])
            state = obs

            for i in range(self.moves_per_epoch):
                actions = self.actions
                action = self.epsilon_greedy(state, actions, self.epsilon)
                #print("Move: {}; action: {}".format(i, actions[action]))

                obs, r, done, _ = env.step(actions[action])
                if train_ep > 10000:
                    print(statep, r)
                    cv2.imshow('hehe', obs)
                    cv2.waitKey(0)
                obs, r = self.get_state(obs)
                #statep, stack_frame = self.get_frame(stack_frame, obs)
                #statep = np.reshape(statep, [1, self.frames, 86, 86, 1])
                statep = obs
                score += r

                self.memory.append((state, action, r, statep))
                
                state = statep

                if train_ep > 0:
                    self.replay()

            print("Epoch: {}; Score: {}; Epsilon: {}".format(train_ep, score, self.epsilon))
            all_scores.append(score)
            if train_ep % 200 == 0:
                self.model.save('configs/model.h5')
                print("Mean score for the last 200 epochs: {}".format(np.average(all_scores[:-200])))

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        gray[gray > 0] = 255
        return gray

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
            return np.array([min_cols, max_cols, min_o_cols, max_o_cols, max_o_rows]), r
        else:
            return np.array([min_cols, max_cols, 0, 0, observation.shape[0] + 1]), 0

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def act(self, observation: np.ndarray):
        """
        :param observation: numpy array of shape (width, height, 3) *defined in config file
        :return: int between 0 and max_action
        """
        state, _ = self.get_state(observation)
        actions = self.actions
        action = self.epsilon_greedy(state, actions, 0)
        return actions[action]

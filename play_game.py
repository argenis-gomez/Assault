import gym
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time


class DQN(keras.Model):

    def __init__(self, n_actions):
        super(DQN, self).__init__()

        self.layer1 = layers.Conv2D(16, 5, strides=2, activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.layer2 = layers.Conv2D(16, 5, strides=2, activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.layer3 = layers.Conv2D(32, 5, strides=2, activation="relu")
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.layer4 = layers.Dense(512, activation="relu")
        self.action = layers.Dense(n_actions, activation="linear")

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.layer4(x)
        return self.action(x)


def take_action(state, EPSILON):
    sample = random.random()
    if sample > (1-EPSILON):
        action = env.action_space.sample()
    else:
        action = np.argmax(model(state))
    next_state, reward, done, _ = env.step(action)
    return next_state/255., reward, done, action


env = gym.make('Assault-v0')

EPISODES = 10
n_actions = env.action_space.n

model = DQN(n_actions)
model.build((None, 250, 160, 3))
model.load_weights('model/model_weights')

for episode in range(1, EPISODES + 1):
    start = time.time()
    state = env.reset() / 255.
    done = False
    episode_reward = 0

    while not done:
        env.render()
        next_state, reward, done, action = take_action(np.expand_dims(state, 0), 0)
        episode_reward += reward
        state = next_state

    print(f"Episodio: {episode} - Recompensa: {episode_reward} - {(time.time()-start):2.2f} segs")

env.close()

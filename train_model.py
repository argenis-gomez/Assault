import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple, deque
import time


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


env = gym.make('Assault-v0')

BATCH_SIZE = 128
GAMMA = 0.999
EPSILON = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 2000
TARGET_UPDATE = 5
EPISODES = 30
n_actions = env.action_space.n

model = DQN(n_actions)
model_target = DQN(n_actions)
memory = ReplayMemory(10000)

optimizer = keras.optimizers.Adam(lr=2.5e-4, clipnorm=1.0)
loss_function = keras.losses.Huber()


def take_action(state, EPSILON):
    sample = random.random()
    if sample > (1-EPSILON):
        action = env.action_space.sample()
    else:
        action = np.argmax(model(state))
    next_state, reward, done, _ = env.step(action)
    return next_state/255., reward, done, action


def optimize_model():
    if memory.__len__() < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = memory.transition(*zip(*transitions))

    state_batch = np.array(batch.state)
    action_batch = np.array(batch.action)
    next_state_batch = np.array(batch.next_state)
    rewad_batch = np.array(batch.reward)
    done_batch = np.array(batch.done, dtype=np.int8)

    future_rewards = model_target(next_state_batch)
    target = rewad_batch + GAMMA * tf.reduce_max(future_rewards, axis=-1) * (1 - done_batch)

    action_mask = tf.one_hot(action_batch, n_actions)

    with tf.GradientTape() as tape:
        q_values = model(state_batch)
        q_action = tf.reduce_sum(tf.multiply(q_values, action_mask), axis=-1)
        loss = loss_function(target, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    model.save_weights('/content/drive/MyDrive/Assault/model/model_weights')


for episode in range(1, EPISODES + 1):
    start = time.time()
    state = env.reset() / 255.
    done = False
    episode_reward = 0

    while not done:
        # env.render()
        print(".", end="")
        next_state, reward, done, action = take_action(np.expand_dims(state, 0), EPSILON)
        memory.push(state, action, next_state, reward, done)
        optimize_model()
        EPSILON -= EPSILON / EPSILON_DECAY
        EPSILON = max(EPSILON, EPSILON_END)
        episode_reward += reward
        state = next_state

    print(f"\nEpisodio: {episode} - Recompensa: {episode_reward} - {(time.time()-start)/60:2.2f} min")

    if episode % TARGET_UPDATE == 0:
        print("TARGET MODEL UPDATED...")
        model_target.set_weights(model.get_weights())

env.close()

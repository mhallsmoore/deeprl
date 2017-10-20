import collections
import random

import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np


class DeepQNetworkAgent(object):
    def __init__(
        self, state_size, action_size, memory_size, gamma, 
        epsilon, epsilon_min, epsilon_decay, learning_rate,
        layer_size
    ):
        # The size of the state space (four) ??
        self.state_size = state_size

        # The size of the action space (two)
        # for "left" and "right"
        self.action_size = action_size

        # A double-ended queue (deque) representing a
        # maximum amount of stored experiences
        self.memory = collections.deque(maxlen=memory_size)

        # Discount rate used to discount future rewards
        self.gamme = gamma

        # Exploration rate for the epsilon-greedy
        # exploration algorithm (is decayed over time)
        self.epsilon = epsilon

        # The minimum exploration rate once decay occurs
        self.epsilon_min = epsilon_min

        # The decay rate for the epsilon value in the EG
        # algorithm to eventually become epsilon_min
        self.epsilon_decay = epsilon_decay

        # The learning rate of the optimiser used to
        # fit the neural network
        self.learning_rate = learning_rate

        # Dense layer size of the Deep Q Network
        self.layer_size = layer_size

        # Construct the Deep Q Network
        self.network = self._construct_network()

    def _construct_network(self):
        """
        Construct the Deep Q Network using Keras
        """
        # Create a feedforward/multi-layer-perceptron
        # (MLP) neural network
        dqn = Sequential()

        # Add a dense hidden layer layer
        dqn.add(
            Dense(
                self.layer_size,
                input_dim=self.state_size,
                activation='relu'
            )
        )

        # Add a second hidden layer of equal size
        # to the first hidden layer
        dqn.add(
            Dense(
                self.layer_size,
                activation='relu'
            )
        )

        # Add an output layer with same size as the
        # action space, which in this instance is
        # two - "left" and "right"
        dqn.add(
            Dense(
                self.action_size,
                activation='linear'
            )
        )

        # Compile the Deep Q Network to use Mean Squared Error
        # as a loss metric, with the Adaptive Moment Estimator
        # (Adam) stochastic gradient descent optimiser
        dqn.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )
        return dqn

    def memorise(
        self, state, action,
        reward, next_state, done
    ):
        """
        Memorise an 'experience' into the deque object
        for later use in memory 'replay'.
        """
        self.memory.append(
            state, action, reward,
            next_state, done
        )

    def act_on_state(self, state):
        """
        Use the epsilon-greedy algorithm to carry out
        an action based on the provided state. With prob
        less than epsilon, choose a random action. With
        prob (1-epsilon) choose the action that maximises
        the expected reward.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            action_rewards = self.network.predict(state)
            return np.argmax(action_rewards[0])

    def experience_replay(self, batch_size):
        """
        TODO: Write a verbose docstring here!
        """
        # Randomly sample a batch of memories from the memory deque
        experience_batch = random.sample(self.memory, batch_size)

        # Replay each memory from the batch and extract the information
        for state, action, reward, next_state, done in minibatch:

            # If we're done then Q(s, a) -> r
            target = reward

            # If we're not done then:
            # Q(s, a) -> r + \gamma \text{argmax}_{a} Q(s', a)
            if not done:
                target = (
                    reward + self.gamma *
                    np.amax(self.network.predict(next_state)[0])
                )

            # TODO: Explain these lines
            target_fdr = self.network.predict(state)
            target_fdr[0][action] = target

            # Train the Deep Q Network
            self.network.fit(state, target_fdr, epochs=1, verbose=0)

        # Decay the exploration rate of the epsilon-greedy
        # algorithm for each experience batch
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

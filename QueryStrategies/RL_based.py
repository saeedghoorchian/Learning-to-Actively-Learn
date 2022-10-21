
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import numpy as np


class RLAgent(): #keras.Model
    
    def __init__(self, state_dim, num_actions, Q_hid_dim_1, Q_hid_dim_2, 
                 REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, BATCH_SIZE, MINIBATCH_SIZE, DISCOUNT, UPDATE_TARGET_EVERY):
        
        super(RLAgent, self).__init__()

        self.num_actions = num_actions
        self.state_dim = state_dim
        self.Q_hid_dim_1 = Q_hid_dim_1
        self.Q_hid_dim_2 = Q_hid_dim_2
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE) # An array with last n steps for training
        self.replay_memory_DataColl = deque(maxlen = 100000)
        self.MIN_REPLAY_MEMORY_SIZE = MIN_REPLAY_MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.DISCOUNT = DISCOUNT
        self.target_update_counter = 0 # Used to count when to update target network with main network's weights
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY
        self.model_RL = self.create_model_RL() # Main model_RL
        self.target_model_RL = self.create_model_RL() # Target network
        self.target_model_RL.set_weights(self.model_RL.get_weights())
        
    def create_model_RL(self):
        model_RL = Sequential()
        model_RL.add(Dense(self.Q_hid_dim_1, input_dim = self.state_dim, activation="relu"))
        # model_RL.add(Dense(self.Q_hid_dim_2, activation="relu"))
        model_RL.add(Dense(self.num_actions, activation='linear'))  # ACTION_SPACE_SIZE = how many choices
        
        model_RL.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model_RL

    # Adds step's data to a memory replay array
    def update_replay_memory(self, transition): # (observation space, action, reward, new observation space, done)
        self.replay_memory.append(transition)
    
    def update_replay_memory_DataColl(self, traj): # (observation space, action) #Add Later q_val, 1
        self.replay_memory_DataColl.append(traj)
        return self.replay_memory_DataColl #!

    def train(self, terminal_state, step, saver_count): #terminal state?, step?
        
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = self.replay_memory
#         minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model_RL for Q values
        current_states = np.array([transition[0] for transition in minibatch]) #transition[0] is indeed the Current_StateVec in transition
        current_qs_list = self.model_RL.predict(current_states)

        # Get future states from minibatch, then query NN model_RL for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) #transition[3] is indeed the New_StateVec in transition
        future_qs_list = self.model_RL.predict(new_current_states)
#         future_qs_list = self.target_model_RL.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model_RL.fit(np.array(X), np.array(y), batch_size=self.BATCH_SIZE, verbose=0, shuffle=False) #terminal state?    

        if terminal_state: # Update target network counter every episode
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY: # If counter reaches set value, update target network with weights of main network
            self.target_model_RL.set_weights(self.model_RL.get_weights())
            self.target_update_counter = 0

        if saver_count: 
            self.model_RL.save_weights('drive/MyDrive/Results/RL/RL_trained', save_format='tf')
            print("model is saved!")

    def get_qs(self, state): # Queries main network for Q values given current observation space (environment state)
        return self.model_RL.predict(np.array(state).reshape(-1, *state.shape))[0]
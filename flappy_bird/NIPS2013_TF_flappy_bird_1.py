#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import os
import os.path
import pickle
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import time

FRAME_PER_ACTION = 1
game_name = 'bird_TF_2013_1'  # the name of the game being played for log files
action_size = 2               # number of valid actions
discount_factor = 0.99        # decay rate of past observations
OBSERVE = 100.                # timesteps to observe before training
EXPLORE = 2000000.            # frames over which to anneal epsilon
epsilon_min = 0.00001         # final value of epsilon
epsilon_max = 0.0001          # starting value of epsilon
epsilon_decrement = 0.00001
size_replay_memory = 50000    # number of previous transitions to remember
batch_size = 32               # size of minibatch

model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def build_model():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, action_size])
    b_fc2 = bias_variable([action_size])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # output layer
    output = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, output, h_fc1

def train_model(s, a, y, memory, output, train_step):
    # sample a minibatch to train on
    minibatch = random.sample(memory, batch_size)

    # get the batch variables
    states      = [batch[0] for batch in minibatch]
    actions     = [batch[1] for batch in minibatch]
    rewards     = [batch[2] for batch in minibatch]
    next_states = [batch[3] for batch in minibatch]

    y_batch = []
    output_j1_batch = output.eval(feed_dict = {s : next_states})
    for i in range(0, len(minibatch)):
        done = minibatch[i][4]
        # if done, only equals reward
        if done:
            y_batch.append(rewards[i])
        else:
            y_batch.append(rewards[i] + discount_factor * np.max(output_j1_batch[i]))

    # perform gradient step
    train_step.run(feed_dict = {y : y_batch, a : actions, s : states})
        
def main():
    sess = tf.InteractiveSession()
    s, output, h_fc1 = build_model()

    # define the Loss function
    a = tf.placeholder("float", [None, action_size])
    y = tf.placeholder("float", [None])
    y_prediction = tf.reduce_sum(tf.multiply(output, a), reduction_indices=1)
    Loss = tf.reduce_mean(tf.square(y - y_prediction))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(Loss)
    
    # store the previous observations in replay memory
    # memory = deque()

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    
    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        
        if os.path.isfile(model_path + '/append_memory.pickle'):                        
            with open(model_path + '/append_memory.pickle', 'rb') as f:
                memory = pickle.load(f)

            with open(model_path + '/epsilon_episode.pickle', 'rb') as ggg:
                epsilon, episode = pickle.load(ggg)
                epsilon = 0.001

        print("\n\n Successfully loaded:", checkpoint.model_checkpoint_path,"\n\n")
    else:
        epsilon = epsilon_max
        print("Could not find old network weights")

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(action_size)
    do_nothing[0] = 1
    state, reward_0, done = game_state.frame_step(do_nothing)
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
    stacked_state = np.stack((state, state, state, state), axis=2)

    # start training
    
    time_step = 0
    episode = 0
    progress = ""
    start_time = time.time()
    
    while time.time() - start_time < 5*60:        
        if len(memory) < size_replay_memory:
            progress = "Exploration"            
        else:
            progress = "Training"
                
        episode_step = 0
        done = False
        while not done and episode_step < 1000:
            episode_step += 1
            time_step += 1

            # choose an action epsilon greedily
            Qvalue = output.eval(feed_dict={s : [stacked_state]})[0]
            action = np.zeros([action_size])
            action_index = 0
            if time_step % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(action_size)
                    action[random.randrange(action_size)] = 1
                else:
                    action_index = np.argmax(Qvalue)
                    action[action_index] = 1
            else:
                action[0] = 1 # do nothing

            # run the selected action and observe next state and reward
            next_state, reward, done = game_state.frame_step(action)
            next_state = cv2.cvtColor(cv2.resize(next_state, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, next_state = cv2.threshold(next_state, 1, 255, cv2.THRESH_BINARY)
            next_state = np.reshape(next_state, (80, 80, 1))
            #stacked_next_state = np.append(next_state, stacked_state[:,:,1:], axis = 2)
            stacked_next_state = np.append(next_state, stacked_state[:, :, :3], axis=2)

            # store the transition in memory
            memory.append((stacked_state, action, reward, stacked_next_state, done))
            if len(memory) > size_replay_memory:
                memory.popleft()

            # only train if done observing
            if progress == "Training":
                train_model(s, a, y, memory, output, train_step)
                
                # scale down epsilon
                if epsilon > epsilon_min:
                    epsilon -= epsilon_decrement

            # update the old values
            stacked_state = stacked_next_state
            
            if done or episode_step == 1000:
                episode += 1
                print("Episode :{:>5}".format(episode), "/ Episode step :{:>4}".format(episode_step), "/ Progress :", progress, \
                      "/ Epsilon :{:>2.6f}".format(epsilon), "/ Memory size :{:>5}".format(len(memory)))
                break
            
    saver.save(sess, model_path + "/model.ckpt")
    with open(model_path + '/append_memory.pickle', 'wb') as f:
        pickle.dump(memory, f)
        
    save_object = (epsilon, episode) 
    with open(model_path + '/epsilon_episode.pickle', 'wb') as ggg:
        pickle.dump(save_object, ggg)
    print("\n\n Now we save model \n\n")
    sys.exit()

if __name__ == "__main__":
    main()

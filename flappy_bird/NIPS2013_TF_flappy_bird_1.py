#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
from collections import deque
import cv2
import os
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import time

FRAME_PER_ACTION = 1
game_name = 'bird_dqn_2013_1'   # the name of the game being played for log files
action_size = 2               # number of valid actions
discount_factor = 0.99        # decay rate of past observations
OBSERVE = 100000.             # timesteps to observe before training
EXPLORE = 2000000.            # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001        # final value of epsilon
INITIAL_EPSILON = 0.0001      # starting value of epsilon
size_replay_memory = 50000    # number of previous transitions to remember
batch_size = 32                    # size of minibatch

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
    x_image = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 4) + b_conv1)
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

    return x_image, output, h_fc1

def train_model(s, output, h_fc1, sess):
    # define the Loss function
    action_target = tf.placeholder("float", [None, action_size])
    y_target = tf.placeholder("float", [None])
    
    sess.run(tf.initialize_all_variables())
    
    # saving and loading networks
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("\n\n Successfully loaded:", checkpoint.model_checkpoint_path,"\n\n")
    else:
        print("\n\n Could not find old network weights \n\n")

    y_prediction = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices=1)
    Loss = tf.reduce_mean(tf.square(y_target - y_prediction))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(Loss)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    memory = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(action_size)
    do_nothing[0] = 1
    state, reward, done = game_state.frame_step(do_nothing)
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
    stacked_state = np.stack((state, state, state, state), axis=2)

    # start training
    epsilon = INITIAL_EPSILON
    start_time = time.time()
    time_step = 0
    # while (True):
    while time.time() - start_time < 5*60:
        episode_step = 0        
        while episode_step < 1000:
            episode_step += 1
            # choose an action epsilon greedily
            Q_value = output.eval(feed_dict={s : [stacked_state]})[0]
            action = np.zeros([action_size])
            action_index = 0
            if time_step % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    # print("----------Random Action----------")
                    action_index = random.randrange(action_size)
                    action[random.randrange(action_size)] = 1
                else:
                    action_index = np.argmax(Q_value)
                    action[action_index] = 1
            else:
                action[0] = 1 # do nothing

            # scale down epsilon
            if epsilon > FINAL_EPSILON and time_step > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observe next state and reward
            next_state_colored, reward, done = game_state.frame_step(action)
            next_state = cv2.cvtColor(cv2.resize(next_state_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, next_state = cv2.threshold(next_state, 1, 255, cv2.THRESH_BINARY)
            next_state = np.reshape(next_state, (80, 80, 1))
            #stacked_next_state = np.append(next_state, stacked_state[:,:,1:], axis = 2)
            stacked_next_state = np.append(next_state, stacked_state[:, :, :3], axis=2)

            # store the transition in memory
            memory.append((stacked_state, action, reward, stacked_next_state, done))
            if len(memory) > size_replay_memory:
                memory.popleft()

            # only train if done observing
            if time_step > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(memory, batch_size)

                # get the batch variables
                states      = [batch[0] for batch in minibatch]
                actions     = [batch[1] for batch in minibatch]
                rewards     = [batch[2] for batch in minibatch]
                next_states = [batch[3] for batch in minibatch]

                y_array = []
                Q_array = output.eval(feed_dict = {s : next_states})
                for i in range(0, len(minibatch)):
                    done = minibatch[i][4]
                    # if done, only equals reward
                    if done:
                        y_array.append(rewards[i])
                    else:
                        y_array.append(rewards[i] + discount_factor * np.max(Q_array[i]))

                # perform gradient step
                train_step.run(feed_dict = { y : y_array, a : actions, s : states})

            # update the old values
            stacked_state = stacked_next_state
            time_step += 1

            # print info
            progress = ""
            if time_step <= OBSERVE:
                progress = "observe"
            elif time_step > OBSERVE and time_step <= OBSERVE + EXPLORE:
                progress = "explore"
            else:
                progress = "train"

            if episode_step % 500 == 0:
                print("TIMESTEP", time_step, "/ Progress", progress, \
                    "/ EPSILON", epsilon)

    # save progress every 10000 iterations
    print("\n\n Now we save model \n\n")
    saver.save(sess, model_path + "/", global_step = time_step)
        
def main():
    sess = tf.InteractiveSession()
    s, output, h_fc1 = build_model()
    train_model(s, output, h_fc1, sess)

if __name__ == '__main__':
    main()
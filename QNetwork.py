import numpy as np
import gym
import tensorflow as tf


env = gym.make( 'FrozenLake-v0')

action_count = env.action_space.n
state_count = env.observation_space.n

episodes = 2000




X = tf.placeholder( tf.float32, [ 1, state_count ])
W = tf.Variable( tf.random_uniform())
Y = tf.placeholder( tf.float32, [ 1, action_count ])

Qpred = X*W
learning_rate = 0.1
discount = 0.99
loss = tf.reduce_sum( tf.square( Qpred-Y ) )
train = tf.train.GradientDescentOptimizer( learning_rate ).minimize( loss )

reward_list = []
rewardsum = 0
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run( init )
    for i in range( episodes):
        done = False
        s = env.reset()
        e = 1./(50.0/i+10.)
        rewardsum = 0
        while True:
            Q = tf.run( Qpred, { X : tf.one_hot( s, 16)})
            if np.random.rand() < e:
                s = env.action_space.sample()
            else:
                s = np.argmax( Q )

            s1, done, reward, _ = env.step( s )

            if done:
                Y[0:s] = reward
                break
            else:
                Y[0:s] = reward + discount*np.argmax( tf.run( Qpred, { X : tf.one_hot( s1, 16 ) } ) )

            sess.run( train )

            s = s1

            rewardsum += reward
        reward_list.append( rewardsum)


print( "Success ratio:"+sum(reward_list)*100.0/episodes)






import tensorflow as tf
import gym

import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

def one_hot(x):
    return np.identity( 16 )[x:x+1]


def main():

    register(
        id='FrozenLake-v3',
        entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery' : False }
    )

    env = gym.make( 'FrozenLake-v3')
    input_size = env.observation_space.n
    output_size = env.action_space.n
    learningrate = 0.1


    X = tf.placeholder( dtype=tf.float32, shape=[1, input_size])

    W = tf.Variable( tf.random_uniform( [ input_size, output_size], 0, 0.01))

    Qpred = tf.matmul( X, W )

    Y = tf.placeholder( shape=[1, output_size], dtype=tf.float32)

    loss = tf.reduce_sum( tf.square(Y-Qpred))

    train = tf.train.GradientDescentOptimizer( learningrate).minimize( loss )

    discount = 0.99
    num_episodes = 2000
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run( init )

        rList = []


        for i in range( num_episodes ):
            s = env.reset()
            e = 1. / ((i/50) + 10)
            rAll = 0
            done =False
            local_loss = []

            while not done:
                Qs = sess.run( Qpred, feed_dict={X:one_hot(s)})

                if np.random.rand(1) < e:
                    a = env.action_space.sample()
                else:
                    a = np.argmax( Qs )

                s1, reward, done, _ = env.step( a )
                if done:
                    Qs[0, a] = reward
                else:
                    Qs1 = sess.run( Qpred, feed_dict={X:one_hot(s1)})
                    Qs[0, a] = reward + discount*np.max(Qs1)

                sess.run( train, feed_dict={X:one_hot(s), Y:Qs})

                rAll += reward

                s = s1

            rList.append( rAll )


    print( "Percentage "+ str(sum(rList)/num_episodes))

    plt.bar( range(len(rList)), rList, color="blue")
    plt.show()


if __name__ == '__main__':
    main()
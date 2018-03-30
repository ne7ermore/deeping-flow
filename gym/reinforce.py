import argparse

parser = argparse.ArgumentParser(description='gym reinforce')

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--hidden_dim', type=int, default=128)

args = parser.parse_args()

# ##############################################################################
# CartPole-v0
################################################################################

import gym
env = gym.make('CartPole-v0')
env.seed(args.seed)

IN_DIM = env.observation_space.shape[0]
OUT_DIM = env.action_space.n

# ##############################################################################
# Model
################################################################################

import tensorflow as tf
import numpy as np

tf.set_random_seed(args.seed)


class Policy(object):
    def __init__(self, in_dim, out_dim, h_dim):
        with tf.variable_scope('init_variables'):
            self.state = tf.placeholder(
                tf.float32, [None, in_dim], name="state")
            self.rewards = tf.placeholder(
                tf.float32, [None], name="rewards")
            self.selected_actions = tf.placeholder(
                tf.float32, [None], name="actions")

        self.saved_props = []

        with tf.variable_scope('init_layers'):
            lr1 = tf.keras.layers.Dense(h_dim)
            lr2 = tf.keras.layers.Dense(out_dim)

        hidden = tf.nn.relu(lr1(self.state))
        props = tf.nn.softmax(lr2(hidden))

        dist = tf.distributions.Categorical(props)
        self.action = dist.sample()
        self.log_scores = dist.log_prob(self.selected_actions)

        with tf.variable_scope('loss'):
            loss = tf.reduce_sum(-self.log_scores * self.rewards)
            self.train_op = tf.train.AdamOptimizer(args.lr).minimize(loss)


# ##############################################################################
# Train
################################################################################
from itertools import count

policy = Policy(IN_DIM, OUT_DIM, args.hidden_dim)


def main():
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sv = tf.train.Supervisor(saver=saver, summary_op=None)
    summary_writer = sv.summary_writer

    with sv.managed_session(config=config) as sess:
        for _ in count(1):
            state = env.reset()
            env.render()
            R, states, policy_rewards, rewards, actions = 0, [
                state], [], [], []

            for step in range(10000):
                action = sess.run(
                    [policy.action], feed_dict={policy.state: [state]})[0][0]
                state, reward, done, _ = env.step(action)
                policy_rewards.append(reward)
                actions.append(action)
                if done:
                    break
                states.append(state)

            for r in policy_rewards[::-1]:
                R = r + args.gamma * R
                rewards.insert(0, R)

            rewards = np.asarray(rewards)
            rewards = (rewards - rewards.mean()) / \
                (rewards.std() + np.finfo(np.float32).eps)
            feed_dict = {
                policy.state: np.asarray(states),
                policy.rewards: rewards,
                policy.selected_actions: np.asarray(actions),
            }

            sess.run([policy.train_op], feed_dict)

            print("This time length - {}".format(step))


if __name__ == '__main__':
    main()

import argparse

parser = argparse.ArgumentParser(description='gym reinforce')

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--render', action='store_true')

args = parser.parse_args()

# ##############################################################################
# CartPole-v0
################################################################################

import gym
env = gym.make('CartPole-v0').unwrapped
env.seed(args.seed)

IN_DIM = env.observation_space.shape[0]
OUT_DIM = env.action_space.n

# ##############################################################################
# Model
################################################################################

import tensorflow as tf
import numpy as np

tf.set_random_seed(args.seed)


class ActorCritic(object):
    def __init__(self, in_dim, out_dim, h_dim):
        with tf.variable_scope('init_variables'):
            self.state = tf.placeholder(
                tf.float32, [None, in_dim], name="state")
            self.rewards = tf.placeholder(
                tf.float32, [None], name="rewards")
            self.selected_actions = tf.placeholder(
                tf.float32, [None], name="actions")
            self.td_error = tf.placeholder(
                tf.float32, [None], name="td_error")

        with tf.variable_scope('init_layers'):
            h_layer = tf.keras.layers.Dense(h_dim, activation=tf.nn.relu)
            action_layer = tf.keras.layers.Dense(
                out_dim, activation=tf.nn.softmax)
            value_layer = tf.keras.layers.Dense(1)

        with tf.variable_scope('init_graph'):
            hidden = h_layer(self.state)
            props = action_layer(hidden)
            self.value = tf.reshape(value_layer(hidden), [-1])

            dist = tf.distributions.Categorical(props)
            self.action = dist.sample()
            self.log_scores = dist.log_prob(self.selected_actions)

        with tf.variable_scope('loss'):
            value_loss = self._smooth_l1_loss(self.value, self.rewards)
            action_loss = -tf.reduce_sum(self.log_scores * self.td_error)

            self.train_op = tf.train.AdamOptimizer(
                args.lr).minimize(value_loss + action_loss)

    def _smooth_l1_loss(self, value, rewards):
        thres = tf.constant(1, dtype=tf.float32)
        mae = tf.abs(value - rewards)
        loss = tf.keras.backend.switch(tf.greater(
            mae, thres), (mae - 0.5), 0.5 * tf.pow(mae, 2))

        return tf.reduce_sum(loss)


# ##############################################################################
# Train
################################################################################
from itertools import count

ac = ActorCritic(IN_DIM, OUT_DIM, args.hidden_dim)


def train():
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    more = 0

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        for epoch in count(1):
            state = env.reset()
            if args.render:
                env.render()
            states, policy_rewards, actions, values = [
                state], [], [], []

            for step in range(10000):
                action, value = sess.run([ac.action, ac.value], feed_dict={
                    ac.state: [state]})
                action, value = action[0], value[0]
                state, reward, done, _ = env.step(action)
                policy_rewards.append(reward)
                actions.append(action)
                values.append(value)
                if done:
                    break
                states.append(state)

            R, rewards = 0, []
            for r in policy_rewards[::-1]:
                R = r + args.gamma * R
                rewards.insert(0, R)

            rewards = np.asarray(rewards)
            rewards = (rewards - rewards.mean()) / \
                (rewards.std() + np.finfo(np.float32).eps)
            values = np.asarray(values)

            feed_dict = {
                ac.state: np.asarray(states),
                ac.rewards: rewards,
                ac.selected_actions: np.asarray(actions),
                ac.td_error: (rewards - values),
            }
            sess.run([ac.train_op], feed_dict)

            if more < step:
                print('Epoch {}\tlength: {:5d}\t'.format(epoch, step))
                more = step


if __name__ == '__main__':
    train()

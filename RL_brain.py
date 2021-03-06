"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            input_length,
            output_length,
            output_shape,
            action_dim,
            action_space,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=50000,
            batch_size=50000,
            e_greedy_increment=None,
            output_graph=False,
            use_gdm=False,
            sess=None
    ):
        self.input_length = input_length
        self.output_length = output_length
        self.output_shape = output_shape
        self.action_dim = action_dim
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.replace_flag = False
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.input_length * 2 + 1 + self.action_dim))

        self.cost_his = []
        self.reward_his = []
        self.memory_counter = 0

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        self._build_net()
        
        #if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            #tf.summary.FileWriter("logs/", self.sess.graph)

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.input_length], name='s')
        self.q_target = tf.placeholder(tf.float32, [None]+self.output_shape, name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.input_length, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.input_length, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('ladd'):
                wadd = tf.get_variable('wadd', [n_l1,n_l1], initializer=w_initializer, collections=c_names)
                badd = tf.get_variable('badd', [1, n_l1], initializer=b_initializer, collections=c_names)
                ladd = tf.nn.relu(tf.matmul(l1, wadd) + badd)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.output_length], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.output_length], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.reshape(tf.matmul(ladd, w2) + b2, [-1]+self.output_shape)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.input_length], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.input_length, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('ladd'):
                wadd = tf.get_variable('wadd', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                badd = tf.get_variable('badd', [1, n_l1], initializer=b_initializer, collections=c_names)
                ladd = tf.nn.relu(tf.matmul(l1, wadd) + badd)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.output_length], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.output_length], initializer=b_initializer, collections=c_names)
                self.q_next = tf.reshape(tf.matmul(ladd, w2) + b2, [-1]+self.output_shape)
        
        
        # consist of [target_net, evaluate_net]
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess.run(tf.global_variables_initializer())

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, a, r, s_))
        #print(transition)

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def value_func(self, observation):
        observation = observation[np.newaxis, :]
        v = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        return v

    def learn(self, flag):
        if self.memory_counter == 0:
            return
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_flag = True
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
    
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.input_length:],  # fixed params
                self.s: batch_memory[:, :self.input_length],  # newest params
            })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        #print("q_target:",q_target.shape)

        #batch_index = np.arange(self.batch_size, dtype=np.int32)

        eval_act_index = batch_memory[:, self.input_length:self.input_length+self.action_dim].astype(int)
        #print("eval_act_index:", eval_act_index.shape)

        reward = batch_memory[:, self.input_length + self.action_dim].\
            repeat(self.output_length).reshape(self.batch_size, self.output_length)  #  repeat max_coop*n_actions
        #print("reward:", reward.shape)

        m = np.zeros(shape=(self.batch_size, self.action_dim))  # batch_size,max_coop
        for i in range(self.batch_size):
            for j in range(self.action_dim):
                m[i, j] = max(q_next[i, j])  # n_actions
        #print("m:",m.shape)

        for i in range(self.batch_size):  # batch_size
            real_coop_count = 0
            for j in range(self.action_dim):
                if eval_act_index[i, j] == -1:
                    break
                real_coop_count = real_coop_count + 1
            for j in range(real_coop_count):  # actual_coop
                q_target[i, j, eval_act_index[i, j]] = reward[i, j] + self.gamma * m[i, j]
        #print("q_target:", q_target.shape,q_target)
        #q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.input_length],
                                                self.q_target: q_target})
        #print(self.cost)
        #print("cost:", self.cost, '\n')
        #if flag is True:
            #self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('episode')
        plt.savefig('cost.png')
        #plt.show()
        plt.clf()

    def plot_reward(self):
        plt.plot(np.arange(len(self.reward_his)), self.reward_his)
        plt.ylabel('reward')
        plt.xlabel('episode')
        plt.savefig('reward.png')
        #plt.show()
        plt.clf()

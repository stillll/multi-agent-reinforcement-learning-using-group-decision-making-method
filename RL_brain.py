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

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            max_coop,
            n_agents,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.max_coop = max_coop
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, max_coop * n_features * 2 + 1 + max_coop))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features*self.max_coop], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions*self.max_coop], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features*self.max_coop, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('ladd'):
                wadd = tf.get_variable('wadd', [n_l1,n_l1], initializer=w_initializer, collections=c_names)
                badd = tf.get_variable('badd', [1, n_l1], initializer=b_initializer, collections=c_names)
                ladd = tf.nn.relu(tf.matmul(l1, wadd) + badd)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions*self.max_coop], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions*self.max_coop], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(ladd, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features*self.max_coop], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features*self.max_coop, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('ladd'):
                wadd = tf.get_variable('wadd', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                badd = tf.get_variable('badd', [1, n_l1], initializer=b_initializer, collections=c_names)
                ladd = tf.nn.relu(tf.matmul(l1, wadd) + badd)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions*self.max_coop], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions*self.max_coop], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(ladd, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, a, r, s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            #print("actions_value:",actions_value)
            action = np.argmax(actions_value[:, :self.n_actions])
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def value_func(self, observation):
        observation = observation[np.newaxis, :]
        v = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        return v

    def coop_set_and_coop_state(self, agent_id, env_s, pow_set):
        a_value = -99999999  # average value
        coop_set = np.array([agent_id])
        coop_state_i = np.array(env_s[agent_id*self.n_features: (agent_id+1)*self.n_features])
        coop_state_i = np.append(coop_state_i, [-2] * (self.max_coop * self.n_features - len(coop_state_i)))
        v_set = []
        soft_max_q = []
        for each in pow_set:  # 'each' is an array, subsets with different lengths
            for each_ in each:  # 'each_' is a array, a specific item in a subset which has specific length
                if each_[0] == agent_id:
                    actual_coop = len(each_)
                    tmp_value = 0
                    start = [a * self.n_features for a in each_]
                    end = [a + self.n_features - 1 for a in start]
                    idx = sorted(start + end)
                    coop_state = env_s[idx]
                    if len(coop_state) < self.n_features * self.max_coop:
                        actual_len = len(coop_state)
                        coop_state = np.append(coop_state, [-2] * (self.max_coop * self.n_features - actual_len))
                    q_values = self.value_func(coop_state)
                    real_q_v = q_values[0, :actual_coop * self.n_actions]
                    v_set_tmp = []  # w_i*q_i for each agent in coop set
                    soft_max_q_tmp = []
                    for i in range(actual_coop):
                        real_q_i = real_q_v[i*self.n_actions: (i+1)*self.n_actions]
                        soft_max_q_i = np.exp(real_q_i)/sum(np.exp(real_q_i))
                        soft_max_q_tmp.append(soft_max_q_i)
                        vi = np.dot(soft_max_q_i, real_q_i)
                        v_set_tmp.append(vi)
                        tmp_value += vi
                    tmp_value /= actual_coop
                    if tmp_value > a_value:
                        coop_set = each_
                        coop_state_i = coop_state
                        soft_max_q = soft_max_q_tmp
                        v_set = v_set_tmp
                        a_value = tmp_value
        for e in soft_max_q:
            for i in range(len(e)):
                e[i] = e[i]*self.epsilon + (1-self.epsilon)/len(e)
        return coop_set, coop_state_i, soft_max_q, v_set, a_value

    def learn(self, flag):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features*self.max_coop:],  # fixed params
                self.s: batch_memory[:, :self.n_features*self.max_coop],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        #print("q_target:",q_target.shape)

        #batch_index = np.arange(self.batch_size, dtype=np.int32)

        eval_act_index = batch_memory[:, self.n_features*self.max_coop:self.n_features*self.max_coop+self.max_coop].astype(int)
        #print("eval_act_index:", eval_act_index.shape)

        reward = batch_memory[:, self.max_coop*self.n_features + self.max_coop].\
            repeat(self.max_coop*self.n_actions).reshape(self.batch_size, self.max_coop*self.n_actions)  #  repeat max_coop*n_actions
        #print("reward:", reward.shape)

        m = np.zeros(shape=(self.batch_size, self.max_coop))  # batch_size,max_coop
        for i in range(self.batch_size):
            for j in range(self.max_coop):
                m[i, j] = max(q_next[i, self.n_actions*j:self.n_actions*j+self.n_actions])  # n_actions
        #print("m:",m.shape)

        for i in range(self.batch_size):  # batch_size
            for j in range(self.max_coop):  # actual_coop
                q_target[i, self.n_actions * j + eval_act_index[i, j]] = reward[i, j] + self.gamma * m[i, j]
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
                                     feed_dict={self.s: batch_memory[:, :self.n_features*self.max_coop],
                                                self.q_target: q_target})
        #print("cost:", self.cost, '\n')
        if flag is True:
            self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('episode')
        plt.show()
        





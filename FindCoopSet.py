# for agents
#   coop set; -- son of dqn
#   DQN; -- once
#   GDM; -- son of coop set
from RL_brain import DeepQNetwork
import numpy as np
import itertools as itr
import tensorflow as tf
import pdb

class CoopSet(DeepQNetwork):
    def __init__(self,
                 n_agents,
                 n_actions,
                 max_coop,
                 n_features,  # feature length for single agent
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=2000,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False,
                 sess=None,
                 add_len=0):
        DeepQNetwork.__init__(self,
                              input_length=max_coop*n_features+add_len,
                              output_length=max_coop*n_actions,
                              output_shape=[max_coop,n_actions],
                              action_dim=max_coop,
                              action_space=n_actions,
                              learning_rate=learning_rate,
                              reward_decay=reward_decay,
                              e_greedy=e_greedy,
                              replace_target_iter=replace_target_iter,
                              memory_size=memory_size,
                              batch_size=batch_size,
                              e_greedy_increment=e_greedy_increment,
                              output_graph=output_graph,
                              sess=sess)
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_features = n_features
        self.max_coop = max_coop
        self.all_coop_sets = []  # all agents' coop set
        self.all_coop_sets_l = self.all_coop_sets  # all agents' coop set on last step
        self.n_obv = []  # all agents' coop state of this step
        self.store_n_obv = []  # all agents' coop state of this step using last coop set
        #self.env_s_test = []
        #self.__build_net()


    def build_net(self):
        self.test__env_s = tf.placeholder(tf.float32, [self.n_agents*self.n_features], name='test__env_s')
        each_list, actual_coop = self.get_each_list()
        each_coop_state = self.state_completion(each_list, self.test__env_s)
        self._build_net_(each_coop_state)
        self.each_value, soft_max_q = self.get_value(each_list, actual_coop)
        self.select_coop_set_and_Q(each_list,each_coop_state,self.each_value,soft_max_q)


    def get_each_list(self):
        each_list = []
        actual_coop = []
        pow_set = self.pow_set()
        for i in range(self.n_agents):
            for each in pow_set:  # 'each' is an nd_array, subsets with different lengths
                for each_ in each:  # 'each_' is a nd_array, a specific item in a subset which has specific length
                    if each_[0] == i:
                        each_list.append(each_)
                        j = 0
                        while j < len(each_) and each_[j] < self.n_agents:
                            j += 1
                        actual_coop.append(j)
            #each_list.append(each_list_i)
        each_list = np.array(each_list)
        actual_coop = np.array(actual_coop)
        return each_list, actual_coop


    def pow_set(self):
        all_agent_id = np.array(range(self.n_agents))
        p_set = []
        for i in range(1, self.max_coop + 1):
            tmp = np.array(list(itr.permutations(all_agent_id, i)))
            p_set_ = []
            for each_ in tmp:
                while len(each_) < self.max_coop:
                    each_ = np.append(each_, self.n_agents)
                p_set_.append(each_)
            p_set.append(np.array(p_set_))
        return np.array(p_set)


    def state_completion(self, coop_set_list, env_s):
        idx = []
        list_len = len(coop_set_list)
        start = np.dot(coop_set_list,self.n_features)
        end = start + self.n_features
        for j in range(list_len):
            idx_j = np.array([],dtype = np.int32)
            for i in range(len(start[j])):
                idx_i = np.arange(start[j][i],end[j][i])
                idx_j = np.append(idx_j, idx_i)
            idx.append(np.array(idx_j))
        self.idx = np.array(idx)
        id_inUse = tf.reshape(self.idx,[self.idx.shape[0]*self.idx.shape[1]])
        self.env_new = tf.concat([env_s,[-20.]*self.n_features],0)
        coop_state_list_inUse = tf.gather(self.env_new,id_inUse,axis=0)
        coop_state_list = tf.reshape(coop_state_list_inUse,[self.idx.shape[0],self.idx.shape[1]])
        coop_state_list = tf.cast(coop_state_list, dtype=tf.float32)
        return coop_state_list


    def _build_net_(self,ss):
        with tf.variable_scope('run_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['run_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.input_length, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.input_length, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(ss, w1) + b1)

            with tf.variable_scope('ladd'):
                wadd = tf.get_variable('wadd', [n_l1,n_l1], initializer=w_initializer, collections=c_names)
                badd = tf.get_variable('badd', [1, n_l1], initializer=b_initializer, collections=c_names)
                ladd = tf.nn.relu(tf.matmul(l1, wadd) + badd)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.output_length], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.output_length], initializer=b_initializer, collections=c_names)
                self.q_run = tf.reshape(tf.matmul(ladd, w2) + b2, [-1]+self.output_shape)


        r_params = tf.get_collection('run_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_run_op = [tf.assign(r, e) for r, e in zip(r_params, e_params)]

        

        self.sess.run(tf.global_variables_initializer())

    def get_value(self,each_coop_list, actual_coop):
        zd = []
        for i in range(len(actual_coop)):
            zd_ = []
            for j in range(self.max_coop):
                if j < actual_coop[i]:
                    zd_.append([1.]*self.n_actions)
                else:
                    zd_.append([0.]*self.n_actions)
            zd.append(np.array(zd_))
        zd = np.array(zd,dtype= np.float32)
        zd_opp = 1 - zd
        max_add = tf.reduce_max(self.q_run,axis=2)
        max_add = tf.concat([[max_add]]*self.q_run.shape[2],0)
        max_add = tf.transpose(max_add,[1,2,0])
        sotf_q_new = tf.multiply(self.q_run, zd) + tf.multiply(max_add, zd_opp)
        min_soft_q = tf.reduce_min(sotf_q_new,axis=2)
        min_soft_q = tf.concat([[min_soft_q]]*self.q_run.shape[2],0)
        min_soft_q = tf.transpose(min_soft_q,[1,2,0])
        soft_max_q = sotf_q_new - min_soft_q + 0.01
        soft_max_q = tf.multiply(soft_max_q,zd)
        max_sum = tf.reduce_sum(soft_max_q,axis=2)
        max_sum = tf.concat([[max_sum]]*self.q_run.shape[2],0)
        max_sum = tf.transpose(max_sum,[1,2,0])
        soft_max_q = tf.div(soft_max_q, max_sum)
        value_ = tf.reduce_sum(tf.multiply(soft_max_q, self.q_run),axis=2)
        value_ = tf.reduce_sum(value_,axis=1)
        #actual_coop_ = tf.cast(actual_coop,dtype=tf.float32)
        #value = tf.div(value_,actual_coop_)
        value = value_
        return value, soft_max_q

    def select_coop_set_and_Q(self,set_list_, state_list_, value_list_,soft_max_q):
        dimen = tf.cast(value_list_.shape[0],tf.int32)
        value_list = tf.reshape(value_list_, [self.n_agents, dimen/self.n_agents])
        value_max = tf.math.argmax(value_list,1)
        value_max = tf.cast(value_max,tf.int32)
        arange_v = np.arange(0,self.n_agents, dtype=np.int32)
        arange_v_base = tf.concat([[dimen/self.n_agents]]*self.n_agents,0)
        arange_v_base = tf.cast(arange_v_base,tf.int32)
        arange_v = tf.multiply(arange_v,arange_v_base)
        self.index_ = tf.add(value_max, arange_v)
        self.value = tf.gather(value_list_,self.index_)
        self.set_list = tf.gather(set_list_,self.index_)
        self.coop_state_list = tf.gather(state_list_,self.index_)
        self.Q = tf.gather(self.q_run,self.index_)
        self.soft_max_q = tf.gather(soft_max_q,self.index_)
        self.idx = tf.gather(self.idx,self.index_)


    def train_CoopSet(self, env, save_path, max_episode):
        self.build_net()
        step = 0
        accident = False
        for episode in range(max_episode):
            env_s = env.reset()  # init env and return env state
            #self.env_s_test = env_s
            #self.__build_net()
            store_cost_flag = True  # if store cost
            counter = 0  # if end episode
            #pdb.set_trace()
            idx,env_new,SL,CSL,QQQ = self.sess.run(\
                [self.idx,self.env_new,self.set_list,self.coop_state_list,self.Q],
                 feed_dict={self.test__env_s: env_s})
            join_act = []
            #print(SL)
            for i in range(self.n_agents):
                join_act.append(self.choose_action(QQQ[i][0]))
            while True:  # one step
                # learn
                step += 1
                # break while loop when end of this episode

                #print("last",last_join_act)
                #w_r_ = w_r  # 上一步的奖励系数
                last_SL = SL
                last_CSL = CSL
                last_join_act = join_act
                last_idx = idx
                try:
                    env_s, reward, done = env.step(join_act)  # 当前步
                    env.render()
                except:
                    accident = True
                    break

                # ==============================================
                idx,env_new,SL,CSL,QQQ = self.sess.run(\
                    [self.idx,self.env_new,self.set_list,self.coop_state_list,self.Q],
                     feed_dict={self.test__env_s: env_s})
                join_act = []
                for i in range(self.n_agents):
                    join_act.append(self.choose_action(QQQ[i][0]))
                #action_ = np.argmax(QQQ, axis= 2)
                #action = np.transpose(action_, [1,0])
                #join_act = action[0]
    
                #store.store_n_transitions(gdm, obv, last_join_act, last_sugg_act, reward, w_r_)  # 上一步到当前步的转移经验
                #store.store_n_transitions(cs, obv, last_join_act, reward)

                #print(join_act)
                #print(QQQ)
                self.store_transitions(last_SL,last_CSL,last_join_act,reward,env_new,last_idx)

                if counter > 300 or done:
                    print(counter)
                    break
                counter += 1
                if step % 5 == 0:
                    #print(value)
                    self.learn(True)
                    self.sess.run(self.replace_run_op)
                    store_cost_flag = False

            # record cumulate rewards once an episode
            if episode % 10 == 0:
                self.test_CoopSet(env)
            if accident:
                break


        print('game over')


        self.plot_cost()
        self.plot_reward()
        self.plot_actions_value()


    def choose_action(self, q_values):
        if np.random.uniform() < self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, self.n_actions)
        return action



    def store_transitions(self,last_SL,last_CSL,last_join_act,reward,env_new,idx):
        for i in range(self.n_agents):
            last_coop_act = np.array(last_join_act+[-1])[last_SL[i]]
            CSL = env_new[idx[i]]
            self.store_transition(last_CSL[i], last_coop_act, reward, CSL)



    def test_CoopSet(self,env):
        print("test_episode", len(self.reward_his))
        env_s = env.reset()
        QQQ = self.sess.run(self.Q, feed_dict={self.test__env_s: env_s})
        join_act = []
        for i in range(self.n_agents):
            join_act.append(np.argmax(QQQ[i][0]))
        step = 0
        accident = False
        cumulate_reward = 0
        while step < 300:
            step += 1
            try:
                env_s, reward, done = env.step(join_act)  # 当前步
                env.render()
            except:
                accident = True
                break
            cumulate_reward = reward + cumulate_reward
            QQQ = self.sess.run(self.Q, feed_dict={self.test__env_s: env_s})
            join_act = []
            for i in range(self.n_agents):
                join_act.append(np.argmax(QQQ[i][0]))
            if done:
                break
        self.reward_his.append(cumulate_reward)
        print("reward:", cumulate_reward)
        #q_action = self.sess.run(self.s_,feed_dict={dadada})
        # we should cope nets for every agents, but now we just use target net as a test.
        
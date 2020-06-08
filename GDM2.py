# for agents
#   coop set; -- son of dqn
#   DQN; -- once
#   GDM; -- son of coop set
import numpy as np
from FindCoopSet import CoopSet
import tensorflow as tf
import pdb

class GDM(CoopSet):
    def __init__(self,
                 n_actions,
                 n_agents,
                 n_features,
                 max_coop,
                 cll_ba=0.5,
                 max_discuss=3
                 ):
        CoopSet.__init__(
            self,
            n_agents=n_agents,
            n_actions=n_actions,
            max_coop=max_coop,
            n_features=n_features,  # feature length for single agent
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=2000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            sess=None,
            add_len=max_coop*n_actions
        )
        self.cll_ba = cll_ba
        self.max_discuss = max_discuss
        self.all_agents_prms = [[] for i in range(self.n_agents)]  # 2-dim list to store numpys
        self.who_give_suggestion = [[] for i in range(self.n_agents)]
        self.av_exp_values = np.zeros(n_agents)  # average expect values of all agents
        self.all_v_set = []  # all agents' v_set, e.g. 0's coop set is [0,1,2], all_v_set[i] = [v0,v1,v2]
        self.all_cl = []  # all agents' cl
        self.all_sugg = [0.25 for i in range(self.n_agents * self.n_actions)]  # all agents' final suggestions
        #self.___build_net()


    def __build_net_(self):
        self.test__env_s = tf.placeholder(tf.float32, [self.n_agents*self.n_features], name='test__env_s')
        self.test__suggestion = tf.placeholder(tf.float32,[self.n_agents*self.n_actions],name='test__suggestion')
        each_list, actual_coop = self.get_each_list()
        each_coop_state = self.state_completion(each_list, self.test__env_s)
        each_suggestion = self.suggestion_completion(each_list, self.test__suggestion)
        each_coop_state = tf.concat([each_coop_state,each_suggestion],1)
        self._build_net_(each_coop_state)
        self.each_value, soft_max_q = self.get_value(each_list, actual_coop)
        self.select_coop_set_and_Q(each_list,each_coop_state,self.each_value,soft_max_q)
        self.idx_s = tf.gather(self.idx_s,self.index_)
        self.comunication()
        self.get_omiga()
        self.get_cll()


    def suggestion_completion(self, coop_set_list, suggestions):
        idx = []
        list_len = len(coop_set_list)
        start = np.dot(coop_set_list,self.n_actions)
        end = start + self.n_actions
        for j in range(list_len):
            idx_j = np.array([],dtype = np.int32)
            for i in range(len(start[j])):
                idx_i = np.arange(start[j][i],end[j][i])
                idx_j = np.append(idx_j, idx_i)
            idx.append(np.array(idx_j))
        self.idx_s = np.array(idx)
        id_inUse = tf.reshape(self.idx_s,[self.idx_s.shape[0]*self.idx_s.shape[1]])
        self.suggestion_new = tf.concat([suggestions,[1./self.n_actions]*self.n_actions],0)
        coop_suggestion_list_inUse = tf.gather(self.suggestion_new,id_inUse,axis=0)
        coop_suggestion_list = tf.reshape(coop_suggestion_list_inUse,[self.idx_s.shape[0],self.idx_s.shape[1]])
        coop_suggestion_list = tf.cast(coop_suggestion_list, dtype=tf.float32)
        return coop_suggestion_list


    def comunication(self):
        alt_planA = tf.concat([[self.soft_max_q]]*self.n_actions,0)
        alt_planA = tf.transpose(alt_planA,[1,2,3,0])
        alt_planB = tf.transpose(alt_planA,[0,1,3,2])
        alt_plan = tf.div(alt_planA, tf.add(alt_planA, alt_planB))
        self.alt_plan = alt_plan
        alt_plan = alt_plan[tf.newaxis,:]
        alt_plan = tf.transpose(alt_plan,[3,4,2,1,0])
        set_mext = tf.one_hot(self.set_list,self.n_agents+1)
        set_mext = tf.transpose(set_mext,[2,0,1])
        set_mext = tf.concat([[set_mext]]*self.n_actions,0)
        set_mext = tf.concat([[set_mext]]*self.n_actions,0)
        set_mext = tf.transpose(set_mext,[0,1,4,2,3])
        self.set_mext = set_mext
        a_value = tf.reduce_min(self.value)
        a_value = tf.concat([[a_value]]*self.n_agents,0)
        a_value = self.value - a_value + 0.01
        a_value = tf.div(a_value,tf.reduce_sum(a_value))
        mul_value = tf.concat([[a_value]]*(self.n_agents+1),0)
        mul_value = tf.concat([[mul_value]]*self.max_coop,0)
        mul_value = tf.concat([[mul_value]]*self.n_actions,0)
        mul_value = tf.concat([[mul_value]]*self.n_actions,0)
        set_mext = tf.multiply(set_mext,mul_value)
        #set_mext_sum = tf.reduce_sum(set_mext,axis=4)
        #set_mext_sum = tf.concat([[set_mext_sum]]*self.n_agents,0)
        #set_mext_sum = tf.transpose(set_mext_sum,[1,2,3,4,0])
        #set_mext = tf.div(set_mext,set_mext_sum)
        rfm_i = tf.matmul(set_mext[:,:,:,:self.n_agents,:],alt_plan)
        rfm_i = tf.transpose(rfm_i,[4,2,3,0,1])
        rfm_i = rfm_i[0]
        rfm_i = tf.reduce_sum(rfm_i,axis=0)
        #omiga = self.set_list
        rfm_base = rfm_i[:,0,0]
        rfm_base = tf.concat([[rfm_base]]*self.n_actions,0)
        rfm_base = tf.concat([[rfm_base]]*self.n_actions,0)
        rfm_base = tf.transpose(rfm_base,[2,1,0])
        rfm_i = tf.div(rfm_i,rfm_base)
        rfm_i = tf.div(rfm_i,2)
        self.rfm_i = rfm_i
        #aaa = self.sess.run(rfm_i, feed_dict={self.test__env_s: self.env_s_test})
        Q_div = tf.div(1.,rfm_i)
        Q_div = tf.reduce_sum(Q_div,axis=2)
        Q_div = Q_div - self.n_actions
        QQQ = tf.div(1.,Q_div)
        QQ_sum = tf.reduce_sum(QQQ,axis=1)
        QQ_sum = tf.concat([[QQ_sum]]*self.n_actions,0)
        QQ_sum = tf.transpose(QQ_sum,[1,0])
        self.QQ = tf.div(QQQ,QQ_sum)
        

    def get_omiga(self):
        #self.alt_plan
        set_mext_ = tf.transpose(self.set_mext,[0,1,2,4,3])
        rfm_add = tf.constant(0.5,shape=[self.n_actions,self.n_actions])
        rfm_ = tf.concat([self.rfm_i,[rfm_add]],0)
        rfm_ = tf.concat([[rfm_]]*self.max_coop,0)[tf.newaxis,:]
        rfm_ = tf.transpose(rfm_,[3,4,1,2,0])
        self.rfm_ = rfm_
        self.set_mext_ = set_mext_
        aim_plan = tf.matmul(set_mext_,rfm_)
        aim_plan = tf.transpose(aim_plan,[4,3,2,0,1])
        self.aim_plan = aim_plan
        xxx = tf.concat([[self.alt_plan],aim_plan],0)
        means1, variance1 = tf.nn.moments(xxx,axes=[0])
        variance2 = tf.math.sqrt(variance1)
        means, variance = tf.nn.moments(variance2,axes=[1,2,3])
        self.omiga = 1. - means*(self.n_actions/(self.n_actions-1.))


    def get_cll(self):
        value_sum = tf.reduce_sum(self.value)
        value_inUse = tf.div(self.value,value_sum)
        cll = tf.multiply(value_inUse,self.omiga)
        self.cll = tf.reduce_sum(cll)
        

    def train_CoopSet(self,env, save_path, max_episode):
        self.__build_net_()
        step = 0
        accident = False
        for episode in range(max_episode):
            env_s = env.reset()  # init env and return env state
            #self.env_s_test = env_s
            #self.__build_net()
            store_cost_flag = True  # if store cost
            counter = 0  # if end episode
            #pdb.set_trace()
            omig,idx,idx_s,env_new,suggestion_new,SL,CSL,QQQ,mid_Q = self.discuss(env_s)
            join_act = []
            print(SL)
            print(omig)
            for i in range(self.n_agents):
                join_act.append(self.choose_action(QQQ[i]))
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
                last_idx_s = idx_s
                last_omiga = omig
                last_midQ = mid_Q
                try:
                    env_s, reward, done = env.step(join_act)  # 当前步
                    env.render()
                except:
                    accident = True
                    break

                # ==============================================
                omig,idx,idx_s,env_new,suggestion_new,SL,CSL,QQQ,mid_Q = self.discuss(env_s)
                join_act = []
                for i in range(self.n_agents):
                    join_act.append(self.choose_action(QQQ[i]))
                #action_ = np.argmax(QQQ, axis= 2)
                #action = np.transpose(action_, [1,0])
                #join_act = action[0]
    
                #store.store_n_transitions(gdm, obv, last_join_act, last_sugg_act, reward, w_r_)  # 上一步到当前步的转移经验
                #store.store_n_transitions(cs, obv, last_join_act, reward)

                self.store_transitions(last_SL,last_CSL,last_midQ,reward,env_new,suggestion_new,last_idx,last_idx_s,last_omiga)

                if counter > 300 or done:
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
        #self.all_coop_sets.append(coop_set)
        #self.n_obv.append(coop_state_i)
       # save model
        #saver = tf.train.Saver()
        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)
        #saver.save(cs.sess, save_path)
        # end of game
        print('game over')
        #if accident is False:
            #env.destroy()

        #if not os.path.exists('data_for_plot'):
        #    os.makedirs('data_for_plot')
        #write_rewards = open('data_for_plot/'+str(cs.n_agents)+'-'+str(cs.max_coop)+'-reward_his.txt', 'w+')
        #for ip in cs.reward_his:
        #    write_rewards.write(str(ip))
        #    write_rewards.write('\n')
        #write_rewards.close()

        #write_costs = open('data_for_plot/'+str(cs.n_agents)+'-'+str(cs.max_coop)+'-cost_his.txt', 'w+')
        #for ip in cs.reward_his:
        #    write_costs.write(str(ip))
        #    write_costs.write('\n')
        #write_costs.close()

        self.plot_cost()
        self.plot_reward()
        self.plot_actions_value()


    def discuss(self,env_s):
        suggestion = [1./self.n_actions]*self.n_actions*self.n_agents
        omig,idx,idx_s,env_new,suggestion_new,SL,CSL,QQQ,mid_Q,cll = self.sess.run(\
            [self.omiga,self.idx,self.idx_s,self.env_new,self.suggestion_new,self.set_list,self.coop_state_list,self.QQ,self.Q,self.cll],
                feed_dict={self.test__env_s: env_s,
                self.test__suggestion: suggestion})
        discuss_time = 1
        while(cll < self.cll_ba and discuss_time < self.max_discuss):
            discuss_time = discuss_time + 1
            suggestion = QQQ.flatten()
            omig,idx,idx_s,env_new,suggestion_new,SL,CSL,QQQ,mid_Q,cll = self.sess.run(\
                [self.omiga,self.idx,self.idx_s,self.env_new,self.suggestion_new,self.set_list,self.coop_state_list,self.QQ,self.Q,self.cll],
                    feed_dict={self.test__env_s: env_s,
                    self.test__suggestion: suggestion})
        return omig,idx,idx_s,env_new,suggestion_new,SL,CSL,QQQ,mid_Q


    def store_transitions(self,last_SL,last_CSL,last_midQ,reward,env_new,suggestion_new,idx,idx_s,omig):
        for i in range(self.n_agents):
            last_coop_act = np.argmax(last_midQ[i],axis=1)
            #last_coop_act = np.array(last_join_act+[-1])[last_SL[i]]
            CSL = env_new[idx[i]]
            SuL = suggestion_new[idx_s[i]]
            CSL = np.append(CSL,SuL)
            self.store_transition(last_CSL[i], last_coop_act, omig[i]*reward, CSL)



    def test_CoopSet(self,env):
        print("test_episode", len(self.reward_his))
        env_s = env.reset()
        suggestion = [1./self.n_actions]*self.n_actions*self.n_agents
        QQQ = self.sess.run(self.QQ, feed_dict={self.test__env_s: env_s,
            self.test__suggestion: suggestion})
        join_act = []
        for i in range(self.n_agents):
            join_act.append(self.choose_action(QQQ[i]))
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
            suggestion = [1./self.n_actions]*self.n_actions*self.n_agents
            QQQ = self.sess.run(self.QQ, feed_dict={self.test__env_s: env_s,
                self.test__suggestion: suggestion})
            join_act = []
            for i in range(self.n_agents):
                join_act.append(self.choose_action(QQQ[i]))
            if done:
                break
        self.reward_his.append(cumulate_reward)
        print("reward:", cumulate_reward)
        #q_action = self.sess.run(self.s_,feed_dict={dadada})
        # we should cope nets for every agents, but now we just use target net as a test.
        
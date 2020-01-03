import numpy as np
from RL_brain import DeepQNetwork
from GDM import GroupDM
import itertools as itr


class MAGDMRL(DeepQNetwork, GroupDM):
    def __init__(self,
                 n_agents,
                 n_actions,
                 max_coop,
                 n_features,  # feature length for single agent
                 cll_ba=0.5,
                 max_discuss=5,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=2000,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False,
                 use_gdm=True,
                 sess=None):
        DeepQNetwork.__init__(self,
                              max_coop*(n_features+n_actions),
                              max_coop*n_actions,
                              max_coop,
                              n_actions,
                              learning_rate,
                              reward_decay,
                              e_greedy,
                              replace_target_iter,
                              memory_size,
                              batch_size,
                              e_greedy_increment,
                              output_graph,
                              use_gdm,
                              sess)
        GroupDM.__init__(self,
                         n_actions,
                         n_agents,
                         max_coop,
                         cll_ba,
                         max_discuss)
        self.n_features = n_features
        self.use_gdm = use_gdm
        self.all_coop_sets = []  # all agents' coop set
        self.all_coop_sets_l = self.all_coop_sets  # all agents' coop set on last step
        self.n_obv = []  # all agents' coop state of this step
        self.av_exp_values = np.zeros(n_agents)  # average expect values of all agents
        self.all_v_set = []  # all agents' v_set, e.g. 0's coop set is [0,1,2], all_v_set[i] = [v0,v1,v2]
        self.all_cl = []  # all agents' cl
        self.all_sugg = [0.25 for i in range(self.n_agents*self.n_actions)]  # all agents' final suggestions
        self.all_sugg

    # new space for gdm operation in one step
    def new_gdm_space(self):
        self.all_agents_prms = [[] for i in range(self.n_agents)]
        self.who_give_suggestion = [[] for i in range(self.n_agents)]
        self.av_exp_values = np.zeros(self.n_agents)
        self.all_v_set = []
        self.all_cl = []

    # new space for operation in one step
    def new_space(self):
        self.all_coop_sets = []
        self.n_obv = []

    # generate pow set
    def pow_set(self):
        all_agent_id = np.array(range(self.n_agents))
        p_set = []
        for i in range(1, self.max_coop + 1):
            p_set.append(np.array(list(itr.permutations(all_agent_id, i))))
        return p_set

    # Fill the state length that does not meet the input conditions to a fixed length
    def state_completion(self, coop_set, env_s):
        #print("1",env_s)
        start = [a * self.n_features for a in coop_set]
        end = [a + self.n_features - 1 for a in start]
        idx = np.array([])
        for i in range(len(start)):
            idx_i = np.arange(start[i], end[i] + 1)
            idx = np.append(idx, idx_i).astype(int)
        coop_state = env_s[idx]
        coop_state = np.append(coop_state, [-2]*(self.max_coop*self.n_features-len(coop_state)))
        #print("2",coop_state)
        if self.use_gdm is True:
            start2 = [a * self.n_actions + self.n_agents * self.n_features for a in coop_set]
            end2 = [a + self.n_actions - 1 for a in start2]
            idx2 = np.array([])
            for i in range(len(start2)):
                idx_i2 = np.arange(start2[i], end2[i] + 1)
                idx2 = np.append(idx2, idx_i2).astype(int)
            coop_state = np.append(coop_state, env_s[idx2])
            coop_state = np.append(coop_state, [0.25] * (self.max_coop*self.n_actions - len(coop_set)*self.n_actions))
        
        return coop_state

    # find coop set and coop state for single agent in one step
    def coop_set_and_coop_state(self, agent_id, env_s):
        pow_set = self.pow_set()
        a_value = -99999999  # average expect value
        coop_set = np.array([agent_id])
        coop_state_i = self.state_completion(coop_set, env_s)
        v_set = []
        soft_max_q = []
        for each in pow_set:  # 'each' is an nd_array, subsets with different lengths
            for each_ in each:  # 'each_' is a nd_array, a specific item in a subset which has specific length
                if each_[0] == agent_id:
                    actual_coop = len(each_)
                    tmp_value = 0
                    coop_state = self.state_completion(each_, env_s)
                    q_values = self.value_func(coop_state)
                    #print("q",q_values)
                    real_q_v = q_values[0, :actual_coop * self.n_actions]
                    v_set_tmp = []  # w_i*q_i for each agent in coop set
                    soft_max_q_tmp = []
                    for i in range(actual_coop):
                        real_q_i = real_q_v[i*self.n_actions: (i+1)*self.n_actions]
                        soft_max_q_i = real_q_i - np.min(real_q_i) + 0.01
                        soft_max_q_i = soft_max_q_i / sum(soft_max_q_i)
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

    def run_model(self, env_s):
        cll = 0  # CLL
        discuss_cnt = 0
        self.all_coop_sets_l = self.all_coop_sets
        env_s = np.append(env_s, [0.25]*self.n_agents*self.n_actions)
        while cll < self.cll_ba and discuss_cnt < self.max_discuss*3:
            self.new_space()
            join_act = []
            sugg_act = []
            if self.use_gdm is True:
                self.new_gdm_space()
                self.all_sugg = []
            # agents produce and store their prms
            for i in range(self.n_agents):
                coop_set, coop_state_i, soft_max_q, v_set, av_v = self.coop_set_and_coop_state(i, env_s)
                self.all_coop_sets.append(coop_set)
                self.n_obv.append(coop_state_i)
                if self.use_gdm is True:
                    self.all_v_set.append(v_set)  # values of agents in coop set
                    self.av_exp_values[i] = av_v
                    for p in range(len(coop_set)):
                        sugg_act.append(np.argmax(soft_max_q[p]))
                        self.prefer_relation_mtx(i, coop_set[p], soft_max_q[p])
                    for j in range(self.max_coop-len(coop_set)):
                        sugg_act.append(-1)

            if self.use_gdm is True:
                wa = self.av_exp_values - np.min(self.av_exp_values) + 0.01
                wa = wa / sum(wa)  # WA for CLL

            # agents get their aggregate prms and choose action by the suggestion
            for i in range(self.n_agents):
                action = self.choose_action(i,self.n_obv[i])
                join_act.append(action)
            if self.use_gdm is False:
                break
            else:
                env_s = np.append(env_s, np.array(self.all_sugg).flatten())

            #print(wa,self.all_cl)
            cll = np.dot(wa, self.all_cl)
            #print("cll",cll)
            discuss_cnt += 1

        # weights for reward assignment
        if self.use_gdm is True:
            w_r = np.array(self.all_cl)
        else:
            w_r = np.ones(self.n_agents)
        #print(join_act)
        return join_act, w_r, sugg_act

    def choose_action(self, i, observation):
        if self.use_gdm is True:
            w_prm = []  # wights for p_r_m s
            for e in self.who_give_suggestion[i]:
                w_prm_i = self.all_v_set[e][list(self.all_coop_sets[e]).index(i)]
                w_prm.append(w_prm_i)
            w_prm = w_prm-np.min(w_prm)+0.01
            w_prm = w_prm / sum(w_prm)
            a_prm = self.aggregate_prms(i, w_prm)

            cl_i = 0  # agent_i's sum of Consensus Level
            """
            for p in range(len(self.who_give_suggestion[i])):
                #print("cli", cl_i)
                cl_i += self.con_level(self.all_agents_prms[i][p], a_prm)
            cl_i /= len(self.who_give_suggestion[i])
            """
            for p in self.all_coop_sets[i]:
                cl_i += self.con_level(self.all_agents_prms[p][list(self.who_give_suggestion[p]).index(i)], a_prm)
            cl_i /= len(self.all_coop_sets[i])
            self.all_cl.append(cl_i)
            sugg = self.a_prm_to_sugg(a_prm)
            self.all_sugg.append(sugg)
            if np.random.uniform() < self.epsilon:
                sugg = list(self.a_prm_to_sugg(a_prm))
                action = sugg.index(max(sugg))
            else:
                idx = list(self.a_prm_to_sugg(a_prm)).index(max(sugg))
                max_ = sugg[idx]
                sugg += max_/(self.n_actions-1)
                sugg[idx] = 0
                action = np.random.choice(range(self.n_actions), p=sugg)
        else:
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

    def store_n_transitions(self, last_obv, last_sugg_act, reward, w_r):
        if self.use_gdm is True:
            l_r = reward * w_r
            #print("l_r:", l_r)
        for i in range(self.n_agents):
            #last_coop_act = np.array(last_join_act)[self.all_coop_sets_l[i]]
            #if len(last_coop_act) < self.max_coop:
                #last_coop_act = np.append(last_coop_act, [-1] * (self.max_coop - len(last_coop_act)))
            last_sugg_coop_act = last_sugg_act[i*self.max_coop:(i+1)*self.max_coop]
            if self.use_gdm is True:
                self.store_transition(last_obv[i], last_sugg_coop_act, l_r[i], self.n_obv[i])
            else:
                self.store_transition(last_obv[i], last_sugg_coop_act, reward, self.n_obv[i])

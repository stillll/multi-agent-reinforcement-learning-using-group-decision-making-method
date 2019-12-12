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
                              max_coop,
                              n_agents,
                              n_actions,
                              n_features,
                              learning_rate,
                              reward_decay,
                              e_greedy,
                              replace_target_iter,
                              memory_size,
                              batch_size,
                              e_greedy_increment,
                              output_graph)
        GroupDM.__init__(self,
                         n_actions,
                         n_agents,
                         max_coop,
                         cll_ba)
        self.use_gdm = use_gdm
        self.all_coop_sets = []  # all agents' coop set
        self.n_obv = []  # all agents' coop state of this step
        self.av_exp_values = np.zeros(n_agents)  # average expect values of all agents
        self.all_v_set = []  # all agents' v_set, e.g. 0's coop set is [0,1,2], all_v_set[i] = [v0,v1,v2]
        self.all_cl = []  # all agents' cl
        if sess is not None:
            DeepQNetwork.sess = sess

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

    # find coop set and coop state for single agent in one step
    def coop_set_and_coop_state(self, agent_id, env_s):
        pow_set = self.pow_set()
        a_value = -99999999  # average expect value
        coop_set = np.array([agent_id])
        coop_state_i = np.array(env_s[agent_id*self.n_features: (agent_id+1)*self.n_features])
        coop_state_i = np.append(coop_state_i, [-2] * (self.max_coop * self.n_features - len(coop_state_i)))
        v_set = []
        soft_max_q = []
        for each in pow_set:  # 'each' is an nd_array, subsets with different lengths
            for each_ in each:  # 'each_' is a nd_array, a specific item in a subset which has specific length
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
                        #print("q:",real_q_i)
                        #soft_max_q_i = np.exp(real_q_i)/sum(np.exp(real_q_i))
                        soft_max_q_i = real_q_i - np.min(real_q_i)
                        soft_max_q_i = soft_max_q_i / sum(soft_max_q_i)
                        #print("s_q:", soft_max_q_i)
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
        self.new_space()
        if self.use_gdm is True:
            self.new_gdm_space()
        # agents produce and store their prms
        for i in range(self.n_agents):
            coop_set, coop_state_i, soft_max_q, v_set, av_v = self.coop_set_and_coop_state(i, env_s)
            self.all_coop_sets.append(coop_set)
            self.n_obv.append(coop_state_i)
            if self.use_gdm is True:
                self.all_v_set.append(v_set)  # values of agents in coop set
                self.av_exp_values[i] = av_v
                for p in range(len(coop_set)):
                    self.prefer_relation_mtx(i, coop_set[p], soft_max_q[p])

        #wa = np.exp(self.av_values) / sum(np.exp(self.av_values))  # WA for CLL
        # agents get their aggregate prms and choose action by the suggestion
        join_act = []
        for i in range(self.n_agents):
            if self.use_gdm is True:
                w_prm = [] # wights for p_r_m s
                for e in self.who_give_suggestion[i]:
                    w_prm_i = self.all_v_set[e][list(self.all_coop_sets[e]).index(i)]
                    w_prm.append(w_prm_i)
                #print("w_prm0:", w_prm)
                w_prm = np.exp(w_prm) / sum(np.exp(w_prm))
                #w_prm = w_prm / sum(w_prm)
                #print("w_prm:",w_prm)

                a_prm = self.aggregate_prms(i, w_prm)

                cl_i = 0  # agent_i's sum of Consensus Level
                for p in range(len(self.who_give_suggestion[i])):
                    cl_i += self.con_level(self.all_agents_prms[i][p], a_prm)
                cl_i /= len(self.who_give_suggestion[i])
                self.all_cl.append(cl_i)
                sugg = self.a_prm_to_sugg(a_prm)
                action = np.random.choice(range(self.n_actions), p=sugg)
                join_act.append(action)
            else:
                action = self.choose_action(self.n_obv[i])
                join_act.append(action)
        # weights for reward assignment
        if self.use_gdm is True:
            #w_r = np.array(self.all_cl) / sum(np.array(self.all_cl))
            #w_r = np.array(self.all_cl)
            w_r = np.ones(self.n_agents)
        else:
            w_r = np.ones(self.n_agents)
        return join_act, w_r

    def store_n_transitions(self, join_act, env_s_, reward, w_r):
        for i in range(self.n_agents):
            start = [a * self.n_features for a in self.all_coop_sets[i]]
            end = [a + self.n_features - 1 for a in start]
            idx = sorted(start + end)
            observation_ = env_s_[idx]
            if len(observation_) < self.max_coop * self.n_features:
                observation_ = np.append(observation_, [-2] * (self.max_coop * self.n_features - len(observation_)))
            coop_act = np.array(join_act)[self.all_coop_sets[i]]
            if len(coop_act) < self.max_coop:
                coop_act = np.append(coop_act, [-1] * (self.max_coop - len(coop_act)))
            if self.use_gdm is True:
                r = reward * w_r
                print("r:",r)
            if self.use_gdm is True:
                self.store_transition(self.n_obv[i], coop_act, r[i], observation_)
            else:
                self.store_transition(self.n_obv[i], coop_act, reward, observation_)
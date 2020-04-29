# for agents
#   coop set; -- son of dqn
#   DQN; -- once
#   GDM; -- son of coop set
import numpy as np
from FindCoopSet import CoopSet


class GDM(CoopSet):
    def __init__(self,
                 n_actions,
                 n_agents,
                 n_features,
                 max_coop,
                 cll_ba,
                 max_discuss
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
            sess=None
        )
        self.cll_ba = cll_ba
        self.max_discuss = max_discuss
        self.all_agents_prms = [[] for i in range(self.n_agents)]  # 2-dim list to store numpys
        self.who_give_suggestion = [[] for i in range(self.n_agents)]
        self.av_exp_values = np.zeros(n_agents)  # average expect values of all agents
        self.all_v_set = []  # all agents' v_set, e.g. 0's coop set is [0,1,2], all_v_set[i] = [v0,v1,v2]
        self.all_cl = []  # all agents' cl
        self.all_sugg = [0.25 for i in range(self.n_agents * self.n_actions)]  # all agents' final suggestions

    def new_gdm_space(self):
        self.all_agents_prms = [[] for i in range(self.n_agents)]
        self.who_give_suggestion = [[] for i in range(self.n_agents)]
        self.av_exp_values = np.zeros(self.n_agents)
        self.all_v_set = []
        self.all_cl = []

    def state_completion(self, coop_set, env_s):
        # print("1",env_s)
        start = [a * self.n_features for a in coop_set]
        end = [a + self.n_features - 1 for a in start]
        idx = np.array([])
        for i in range(len(start)):
            idx_i = np.arange(start[i], end[i] + 1)
            idx = np.append(idx, idx_i).astype(int)
        coop_state = env_s[idx]
        coop_state = np.append(coop_state, [-20] * (self.max_coop * self.n_features - len(coop_state)))
        # print("2",coop_state)
        if self.max_discuss > 0:
            start2 = [a * self.n_actions + self.n_agents * self.n_features for a in coop_set]
            end2 = [a + self.n_actions - 1 for a in start2]
            idx2 = np.array([])
            for i in range(len(start2)):
                idx_i2 = np.arange(start2[i], end2[i] + 1)
                idx2 = np.append(idx2, idx_i2).astype(int)
            coop_state = np.append(coop_state, env_s[idx2])
            coop_state = np.append(coop_state,
                                   [0.25] * (self.max_coop * self.n_actions - len(coop_set) * self.n_actions))
        return coop_state

    # compute preference relation matrix
    def prefer_relation_mtx(self, agent_i, to_agent_j, alt_plan):
        prm = np.zeros(shape=(self.n_actions, self.n_actions))
        for i in range(self.n_actions):
            for j in range(self.n_actions):
                if alt_plan[i] == 0 and alt_plan[j] == 0:
                    prm[i, j] = 0
                else:
                    prm[i, j] = alt_plan[i]/(alt_plan[i]+alt_plan[j])
        self.all_agents_prms[to_agent_j].append(prm)
        self.who_give_suggestion[to_agent_j].append(agent_i)
        return prm

    # compute aggregate preference relation matrix
    def aggregate_prms(self, wights, prms):
        a_prm = np.zeros(shape=(self.n_actions, self.n_actions))
        for i in range(len(prms)):
            a_prm += wights[i] * prms[i]
        return a_prm

    # convert aggregated preference relation matrix to suggestion
    def a_prm_to_sugg(self, a_prm):
        suggestion = []
        for i in range(self.n_actions):
            x = 0
            flag = False
            for j in range(self.n_actions):
                if a_prm[i, j] != 0:
                    x += 1/a_prm[i, j]
                else:
                    flag = True
                    break
            if flag is False:
                x = 1/(x-self.n_actions)
            else:
                x = 0
            suggestion.append(x)
        #print("sg:", suggestion)
        suggestion = suggestion / sum(suggestion)
        #print("sg_:", suggestion)
        return suggestion

    # compute Consensus level between prm and a_prm
    def con_level(self, prm, a_prm):
        c_l = 0
        for i in range(self.n_actions):
            for j in range(self.n_actions):
                if i != j:
                    c_l += abs(prm[i, j]-a_prm[i, j])/(self.n_actions*(self.n_actions-1))
        return 1 - c_l

    def run_gdm(self, coop_set, q_values):
        av_v = 0  # average value
        w_prm = []  # wights for p_r_m s
        prms = []
        for i in range(len(coop_set)):
            q_v_i = q_values[i * self.n_actions: (i + 1) * self.n_actions]
            soft_max_q_i = q_v_i - np.min(q_v_i) + 0.01
            soft_max_q_i = soft_max_q_i / sum(soft_max_q_i)
            vi = np.dot(soft_max_q_i, q_v_i)
            w_prm.append(vi)
            av_v += vi
            prm_i = self.prefer_relation_mtx(i, coop_set[i], soft_max_q_i)
            prms.append(prm_i)
        av_v /= len(coop_set)

        w_prm = w_prm - np.min(w_prm) + 0.01
        w_prm = w_prm / sum(w_prm)
        a_prm = self.aggregate_prms(w_prm, prms)

        cl_i = 0  # agent_i's sum of Consensus Level

        for p in prms:
            cl_i += self.con_level(p, a_prm)
        cl_i /= len(coop_set)
        sugg = self.a_prm_to_sugg(a_prm)
        self.all_sugg.append(sugg)
        return sugg, av_v, cl_i

    def run_model(self, env_s):
        self.new_space()
        self.new_gdm_space()
        join_act = []
        for i in range(self.n_agents):
            coop_set, coop_state_i, q_v = self.coop_set_and_coop_state(i, env_s)


            if len(self.all_coop_sets_l) > 0:
                self.store_n_obv.append(self.state_completion(self.all_coop_sets_l[i], env_s))

            action = self.choose_action(q_v[:self.n_actions])
            join_act.append(action)

        return join_act

    def run_with_dicsuss(self):
        print('test')



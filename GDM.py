import numpy as np


class GroupDM:
    def __init__(self,
                 n_actions,
                 n_agents,
                 max_coop,
                 cll_ba,
                 ):
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.max_coop = max_coop
        self.cll_ba = cll_ba
        self.all_agents_prms = [[] for i in range(self.n_agents)]  # 2-dim list to store numpys
        self.who_give_suggestion = [[] for i in range(self.n_agents)]

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

    # compute aggregate preference relation matrix
    def aggregate_prms(self, agent_i, wights):
        a_prm = np.zeros(shape=(self.n_actions, self.n_actions))
        for i in range(len(self.all_agents_prms[agent_i])):
            a_prm += wights[i] * self.all_agents_prms[agent_i][i]
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
        #print(a_prm)
        #print(suggestion)
        suggestion = np.exp(suggestion) / sum(np.exp(suggestion))
        return suggestion

    # compute Consensus level between prm and a_prm
    def con_level(self, prm, a_prm):
        c_l = 0
        for i in range(self.n_actions):
            for j in range(self.n_actions):
                if i != j:
                    c_l += abs(prm[i, j]-a_prm[i, j])/(self.n_actions*(self.n_actions-1))
        return 1 - c_l


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

    def new_space(self):
        """
        for i in range(len(self.all_agents_prms)):
            for j in range(len(self.all_agents_prms[i])):
                self.all_agents_prms[i].pop()
        for i in range(len(self.who_give_suggestion)):
            for j in range(len(self.who_give_suggestion[i])):
                self.who_give_suggestion[i].pop()
        """
        self.all_agents_prms = [[] for i in range(self.n_agents)]
        self.who_give_suggestion = [[] for i in range(self.n_agents)]

    def prefer_relation_mtx(self, agent_i, to_agent_j, alt_plan):
        prm = np.zeros(shape=(self.n_actions, self.n_actions))
        for i in range(self.n_actions):
            for j in range(self.n_actions):
                prm[i, j] = alt_plan[i]/(alt_plan[i]+alt_plan[j])
        self.all_agents_prms[to_agent_j].append(prm)
        self.who_give_suggestion[to_agent_j].append(agent_i)

    def aggregate_prms(self, agent_i, wights):
        a_prm = np.zeros(shape=(self.n_actions, self.n_actions))
        for i in range(len(self.all_agents_prms[agent_i])):
            a_prm += wights[i] * self.all_agents_prms[agent_i][i]
        return a_prm

    def a_prm_to_sugg(self, a_prm):
        suggestion = []
        for i in range(self.n_actions):
            x = 0
            for j in range(self.n_actions):
                x += 1/a_prm[i, j]
            suggestion.append(1/(x-self.n_actions))
        prob = np.exp(suggestion) / sum(np.exp(suggestion))
        #print(prob)
        return prob

    def con_level(self, prm, a_prm):
        c_l = 0
        for i in range(self.n_actions):
            for j in range(self.n_actions):
                if i != j:
                    c_l += abs(prm[i, j]-a_prm[i, j])/(self.n_actions*(self.n_actions-1))
        return 1 - c_l


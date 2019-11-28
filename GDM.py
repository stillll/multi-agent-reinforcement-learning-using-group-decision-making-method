import numpy as np


class GDM:
    def __init__(self,
                 n_actions,
                 n_agents,
                 max_coop,
                 ):
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.max_coop = max_coop
        sel
        self.prms_for_agent = []

    def prefer_relation_mtx(self, agent_i, to_agent_j, alt_plan):
        dim = len(alt_plan)
        prm = np.zeros(shape=(dim, dim))
        for i in range(dim):
            for j in range(dim):
                prm[i, j] = alt_plan[i]/(alt_plan[i]+alt_plan[j])
        return prm

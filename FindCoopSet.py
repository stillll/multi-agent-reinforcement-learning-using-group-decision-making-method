# for agents
#   coop set; -- son of dqn
#   DQN; -- once
#   GDM; -- son of coop set
from RL_brain import DeepQNetwork
import numpy as np
import itertools as itr


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
                 sess=None):
        DeepQNetwork.__init__(self,
                              input_length=max_coop*n_features,
                              output_length=max_coop * n_actions,
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

    def new_space(self):
        self.all_coop_sets = []
        self.n_obv = []
        self.store_n_obv = []

    def pow_set(self):
        all_agent_id = np.array(range(self.n_agents))
        p_set = []
        for i in range(1, self.max_coop + 1):
            p_set.append(np.array(list(itr.permutations(all_agent_id, i))))
        return p_set

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
        return coop_state

    def coop_set_and_coop_state(self, agent_id, env_s):
        pow_set = self.pow_set()
        a_value = -99999999  # average expect value
        coop_set = np.array([agent_id])
        coop_state_i = self.state_completion(coop_set, env_s)
        q_v = []
        for each in pow_set:  # 'each' is an nd_array, subsets with different lengths
            for each_ in each:  # 'each_' is a nd_array, a specific item in a subset which has specific length
                if each_[0] == agent_id:
                    actual_coop = len(each_)
                    tmp_value = 0
                    #print("env_s:",env_s)
                    coop_state = self.state_completion(each_, env_s)
                    #print("coop_state:", coop_state)
                    q_values = self.value_func(coop_state)
                    # print("q",q_values)
                    real_q_v = q_values[0, :actual_coop * self.n_actions]
                    for i in range(actual_coop):
                        real_q_i = real_q_v[i * self.n_actions: (i + 1) * self.n_actions]
                        soft_max_q_i = real_q_i - np.min(real_q_i) + 0.01
                        soft_max_q_i = soft_max_q_i / sum(soft_max_q_i)
                        vi = np.dot(soft_max_q_i, real_q_i)
                        tmp_value += vi
                    tmp_value /= actual_coop
                    if tmp_value > a_value:
                        coop_set = each_
                        coop_state_i = coop_state
                        q_v = real_q_v
                        a_value = tmp_value
        self.all_coop_sets.append(coop_set)
        self.n_obv.append(coop_state_i)
        return coop_set, coop_state_i, q_v

    def choose_action(self, q_values):
        if np.random.uniform() < self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def run_model(self, env_s):
        self.all_coop_sets_l = self.all_coop_sets
        self.new_space()
        join_act = []
        for i in range(self.n_agents):
            # q_v = [1,2,3,
            #        2,3,4,
            #        4,5,6]
            q_v = self.coop_set_and_coop_state(i, env_s)[2]

            # if len(self.all_coop_sets_l) > 0:
            #     self.store_n_obv.append(self.state_completion(self.all_coop_sets_l[i], env_s))

            action = self.choose_action(q_v[:self.n_actions])
            join_act.append(action)

        return join_act

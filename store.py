import numpy as np


def store_n_transitions(model, last_obv, last_join_act, reward):
    for i in range(model.n_agents):
        last_coop_act = np.array(last_join_act)[model.all_coop_sets_l[i]]
        last_coop_act = np.append(last_coop_act, [-1] * (model.max_coop - len(last_coop_act)))
            # print("-----------------------------------")
            # print(last_obv[i])
            # print(last_coop_act)
            # print(reward)
            # print(model.n_obv[i])
        model.store_transition(last_obv[i], last_coop_act, reward, model.n_obv[i])


def store_n_transitions_gdm(model, last_obv, last_join_act, last_sugg_act, reward, w_r):
    l_r = reward * w_r
        # print("l_r:", l_r)
    '''aaa = False
    if reward > 4:
        aaa = True
        for jjj in range(self.n_agents):
            if self.store_n_obv[jjj][0] == 0 and self.store_n_obv[jjj][1] == 0:
                aaa = False
    for jjj in range(self.n_agents):
        if self.store_n_obv[jjj][0] == 0 and self.store_n_obv[jjj][1] == 0:
            if reward < 4:
                aaa = True
    if aaa:
            pdb.set_trace()'''
    for i in range(model.n_agents):
        if model.max_discuss > 0:
            last_sugg_coop_act = last_sugg_act[i * model.max_coop:(i + 1) * model.max_coop]
            model.store_transition(last_obv[i], last_sugg_coop_act, l_r[i], model.n_obv[i])
        else:
            last_coop_act = np.array(last_join_act)[model.all_coop_sets_l[i]]
            last_coop_act = np.append(last_coop_act, [-1] * (model.max_coop - len(last_coop_act)))
            if model.use_gdm is True:
                if True:  # self.step_test(last_obv[i], last_coop_act, self.store_n_obv[i]):
                    l_r = reward * w_r
                    model.store_transition(last_obv[i], last_coop_act, l_r[i], model.store_n_obv[i])
            else:
                print("-----------------------------------")
                print(last_obv[i])
                print(last_coop_act)
                print(reward)
                print(model.n_obv[i])
                model.store_transition(last_obv[i], last_coop_act, reward, model.n_obv[i])
                if True:  # self.step_test(last_obv[i], last_coop_act, self.store_n_obv[i]):
                    model.store_transition(last_obv[i], last_coop_act, reward, model.store_n_obv[i])
                    if i == 0:
                        model.sum_1 = model.sum_1 + 1
                    elif i == 1:
                        model.sum_2 = model.sum_2 + 1

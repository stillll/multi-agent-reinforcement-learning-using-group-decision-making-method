import tensorflow as tf
import os
import pdb
import numpy as np
import store
import GDM2
import FindCoopSet


def train_model(env, cs, save_path, max_episode):
    step = 0
    cumulate_reward = 0
    accident = False
    #w_r = np.ones(cs.n_agents)
    for episode in range(max_episode):
        if episode % 10 == 0:
            print("train_episode", episode)
        env_s = env.reset()  # init env and return env state
        store_cost_flag = True  # if store cost
        counter = 0  # if end episode
        #cumulate_reward = 0
        join_act = cs.run_model(env_s)

        # ==============================================
        # cll = 0  # CLL
        # discuss_cnt = 0
        # env_s = np.append(env_s, [0.25] * gdm.n_agents * gdm.n_actions)
        # gdm.all_coop_sets_l = gdm.all_coop_sets
        # # bug : gdm=false discuss=true error
        # while cll < gdm.cll_ba and discuss_cnt < gdm.max_discuss * 3:
        #     gdm.new_space()
        #     join_act = []
        #     sugg_act = []
        #     gdm.new_gdm_space()
        #     gdm.all_sugg = []
        #     # agents produce and store their prms
        #     for i in range(gdm.n_agents):
        #
        #         coop_set, coop_state_i, q_v = gdm.coop_set_and_coop_state(i, env_s)
        #         gdm.all_coop_sets.append(coop_set)
        #         gdm.n_obv.append(coop_state_i)
        #
        #         sugg, av_v, cl_i = gdm.run_gdm(coop_set, q_v)
        #         gdm.all_sugg.append(sugg)
        #         gdm.av_exp_values[i] = av_v
        #         gdm.all_cl.append(cl_i)
        #
        #         if len(gdm.all_coop_sets_l) > 0:
        #             gdm.store_n_obv.append(gdm.state_completion(gdm.all_coop_sets_l[i], env_s))
        #
        #         for p in range(len(coop_set)):
        #             sugg_act.append(np.argmax(sugg[p]))
        #         for j in range(gdm.max_coop - len(coop_set)):
        #             sugg_act.append(-1)
        #
        #         action = gdm.choose_action(sugg)
        #         join_act.append(action)
        #
        #     env_s = np.append(env_s[:gdm.n_agents * gdm.n_features], np.array(gdm.all_sugg).flatten())
        #
        #     wa = gdm.av_exp_values - np.min(gdm.av_exp_values) + 0.01
        #     wa = wa / sum(wa)  # WA for CLL
        #     print('test',wa, gdm.all_cl)
        #     cll = np.dot(wa, gdm.all_cl)
        #     discuss_cnt += 1
        # # weights for reward assignment
        # w_r = np.array(gdm.all_cl)
        # ==============================================

        while True:  # one step
            # learn
            step += 1
            # break while loop when end of this episode

            #print("last",last_join_act)
            #w_r_ = w_r  # 上一步的奖励系数
            last_env_s = env_s
            try:
                env_s, reward, done = env.step(join_act)  # 当前步
                env.render()
            except:
                accident = True
                break
            cumulate_reward = reward + cumulate_reward
            last_join_act = join_act  # 上一步的动作
            #last_sugg_act = sugg_act
            obv = cs.n_obv  # 上一步的观察

            # ==============================================
            join_act = cs.run_model(env_s)
            # cll = 0  # CLL
            # discuss_cnt = 0
            # env_s = np.append(env_s, [0.25] * gdm.n_agents * gdm.n_actions)
            # gdm.all_coop_sets_l = gdm.all_coop_sets
            # # bug : gdm=false discuss=true error
            # while cll < gdm.cll_ba and discuss_cnt < gdm.max_discuss * 3:
            #     gdm.new_space()
            #     join_act = []
            #     sugg_act = []
            #     gdm.new_gdm_space()
            #     gdm.all_sugg = []
            #     # agents produce and store their prms
            #     for i in range(gdm.n_agents):
            #
            #         coop_set, coop_state_i, q_v = gdm.coop_set_and_coop_state(i, env_s)
            #         gdm.all_coop_sets.append(coop_set)
            #         gdm.n_obv.append(coop_state_i)
            #
            #         sugg, av_v, cl_i = gdm.run_gdm(coop_set, q_v)
            #         gdm.all_sugg.append(sugg)
            #         gdm.av_exp_values[i] = av_v
            #         gdm.all_cl.append(cl_i)
            #
            #         if len(gdm.all_coop_sets_l) > 0:
            #             gdm.store_n_obv.append(gdm.state_completion(gdm.all_coop_sets_l[i], env_s))
            #
            #         for p in range(len(coop_set)):
            #             sugg_act.append(np.argmax(sugg[p]))
            #         for j in range(gdm.max_coop - len(coop_set)):
            #             sugg_act.append(-1)
            #
            #         action = gdm.choose_action(sugg)
            #         join_act.append(action)
            #
            #     env_s = np.append(env_s[:gdm.n_agents * gdm.n_features], np.array(gdm.all_sugg).flatten())
            #
            #     wa = gdm.av_exp_values - np.min(gdm.av_exp_values) + 0.01
            #     wa = wa / sum(wa)  # WA for CLL
            #     cll = np.dot(wa, gdm.all_cl)
            #     discuss_cnt += 1
            # # weights for reward assignment
            # w_r = np.array(gdm.all_cl)
            # ==============================================

            #store.store_n_transitions(gdm, obv, last_join_act, last_sugg_act, reward, w_r_)  # 上一步到当前步的转移经验
            store.store_n_transitions(cs, obv, last_join_act, reward)
            if counter > 300 or done:
                break
            counter += 1
            if step % 5 == 0:
                cs.learn(True)
                store_cost_flag = False

        # record cumulate rewards once an episode
        if episode % 10 == 0:
            print("reward:", cumulate_reward)
        cs.reward_his.append(cumulate_reward)
        if accident:
            break

    # save model
    saver = tf.train.Saver()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saver.save(cs.sess, save_path)
    # end of game
    print('game over')
    #if accident is False:
        #env.destroy()

    if not os.path.exists('data_for_plot'):
        os.makedirs('data_for_plot')
    write_rewards = open('data_for_plot/'+str(cs.n_agents)+'-'+str(cs.max_coop)+'-reward_his.txt', 'w+')
    for ip in cs.reward_his:
        write_rewards.write(str(ip))
        write_rewards.write('\n')
    write_rewards.close()

    write_costs = open('data_for_plot/'+str(cs.n_agents)+'-'+str(cs.max_coop)+'-cost_his.txt', 'w+')
    for ip in cs.reward_his:
        write_costs.write(str(ip))
        write_costs.write('\n')
    write_costs.close()

    cs.plot_cost()
    cs.plot_reward()
    cs.plot_actions_value()


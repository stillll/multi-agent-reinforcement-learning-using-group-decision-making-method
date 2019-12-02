import itertools as itr
import numpy as np


def run_model(env, dqn, gdm):
    step = 0
    all_agent_id = np.array(range(env.n_agents))
    pow_set = []
    for i in range(1, env.max_coop+1):
        pow_set.append(np.array(list(itr.permutations(all_agent_id, i))))
    reward_his = []
    cumulate_reward = 0
    for episode in range(100):
        # initial observation
        print("episode", episode, '\n')
        env.reset()  # init env
        env_s_ = env.env_s()
        flag = True
        counter = 0
        while True:
            # RL choose action based on observation and suggestion for every agents
            env_s = env_s_
            join_act = []  # all agents join actions for this step
            n_obv = []  # all agents' coop state of this step
            av_values = np.zeros(env.n_agents)  # average values of all agents
            all_coop_sets = []  # all agents' coop set
            for i in range(env.n_agents):  # agents produce their prms and ect.
                """
                coop_set = coop_set(i)
                ob = obv(i,coop_set)
                Q = Q(ob)
                suggestion = GDM(Q)
                choose_action(i, suggestion)
                """
                observation, coop_set, wight, q_v = dqn.coop_set_and_coop_state(i, env_s, pow_set)
                n_obv.append(observation)
                all_coop_sets.append(coop_set)
                av_values[i] = wight
                q_v_ = np.exp(q_v) / sum(np.exp(q_v))  # softmax q values
                for k in range(len(coop_set)):
                    gdm.prefer_relation_mtx(i, coop_set[k], q_v_[k*env.n_actions:(k+1)*env.n_actions])
                #action = RL.choose_action(observation)
                #join_act.append(action)

            cll = 0
            wa = np.exp(av_values)/sum(np.exp(av_values))
            for i in range(env.n_agents):  # agents get their aggregate prms and choose action by the suggestion
                #wights = []  # wights for p_r_m s
                cl_i = 0  # agent_i's sum of Consensus Level
                """
                denominator = 0  # 分母
                for each in gdm.who_give_suggestion[i]:
                    denominator += np.exp(av_values[each])
                for e in gdm.who_give_suggestion[i]:
                    wights.append(np.exp(av_values[e])/denominator)
                """
                wights = np.exp(av_values[gdm.who_give_suggestion[i]])/sum(np.exp(av_values[gdm.who_give_suggestion[i]]))
                a_prm = gdm.aggregate_prms(i, wights)
                for p in range(len(gdm.who_give_suggestion[i])):
                    cl_i += gdm.con_level(gdm.all_agents_prms[i][p], a_prm)
                cll += (cl_i*wa[i])/env.n_actions
                sugg = gdm.a_prm_to_sugg(a_prm)
                action = np.random.choice(range(env.n_actions), p=sugg)
                join_act.append(action)

            # RL take action and get next observation and reward
            reward, done = env.step(join_act)
            r = reward*wa
            cumulate_reward += reward*pow(0.99, episode)
            counter += 1
            if counter > 300:
                break

            # fresh env
            env.render()
            env_s_ = env.env_s()
            for i in range(env.n_agents):
                observation_ = dqn.coop_set_and_coop_state(i, env_s_, pow_set)[0]
                coop_act = np.array(join_act)[coop_set]
                if len(coop_act) < env.max_coop:
                    coop_act = np.append(coop_act, [-1]*(env.max_coop-len(coop_act)))
                dqn.store_transition(n_obv[i], coop_act, r[i], observation_)

            if (step > 200) and (step % 5 == 0):
                dqn.learn(flag)
                flag = False

            # break while loop when end of this episode
            if done:
                break
            step += 1
        print("reward:", cumulate_reward)
        reward_his.append(cumulate_reward)
    # end of game
    print('game over')
    env.destroy()
    plot_reward(reward_his)


def plot_reward(reward_his):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(reward_his)), reward_his)
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()
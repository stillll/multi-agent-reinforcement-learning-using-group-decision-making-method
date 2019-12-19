import tensorflow as tf


def train_model(env, model, save_path, max_episode):
    step = 0
    cumulate_reward = 0
    for episode in range(max_episode):
        print("train_episode", episode)
        env_s = env.reset()  # init env and return env state
        store_cost_flag = True  # if store cost
        counter = 0  # if end episode
        join_act, w_r = model.run_model(env_s)  # 第一步
        env_s_, reward, done = env.step(join_act)  # 第一步
        while True:  # one step
            cumulate_reward = reward + cumulate_reward * 0.99

            # learn
            step += 1
            if (step > 200) and (step % 5 == 0):
                model.learn(store_cost_flag)
                store_cost_flag = False

            # break while loop when end of this episode
            counter += 1
            if counter > 300 or done:
                break

            # fresh env
            env.render()

            obv = model.n_obv  # 上一步的观察
            last_join_act = join_act  # 上一步的动作
            #print("last",last_join_act)
            r = reward  # 上一步的奖励
            w_r_ = w_r  # 上一步的奖励系数

            env_s = env_s_
            join_act, w_r = model.run_model(env_s)  # 当前步
            #print("now",join_act)
            env_s_, reward, done = env.step(join_act)  # 当前步

            model.store_n_transitions(obv, last_join_act, r, w_r_)  # 上一步到当前步的转移经验

        # record cumulate rewards once an episode
        print("reward:", cumulate_reward)
        model.reward_his.append(cumulate_reward)

    # save model
    saver = tf.train.Saver()
    saver.save(model.sess, save_path)
    # end of game
    print('game over')
    env.destroy()

    write_rewards = open('data_for_plot/3-2-reward_his.txt', 'w')
    for ip in model.reward_his:
        write_rewards.write(str(ip))
        write_rewards.write('\n')
    write_rewards.close()

    write_costs = open('data_for_plot/3-2-cost_his.txt', 'w')
    for ip in model.reward_his:
        write_costs.write(str(ip))
        write_costs.write('\n')
    write_costs.close()

    model.plot_cost()
    model.plot_reward()

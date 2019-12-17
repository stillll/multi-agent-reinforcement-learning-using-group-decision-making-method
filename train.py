import tensorflow as tf


def train_model(env, model, save_path, max_episode):
    step = 0
    cumulate_reward = 0
    for episode in range(max_episode):
        print("train_episode", episode)
        env_s = env.reset()  # init env and return env state
        store_cost_flag = True  # if store cost
        counter = 0  # if end episode
        join_act, w_r = model.run_model(env_s)
        env_s_, reward, done = env.step(join_act)
        while True:  # one step
            # fresh env
            env.render()

            # all agents join actions for this step
            obv = model.n_obv
            joa = join_act
            w = w_r
            join_act, w_r = model.run_model(env_s)

            # take action and get next env state and reward
            r = reward
            env_s_, reward, done = env.step(join_act)

            model.store_n_transitions(joa, obv,r, w)

            cumulate_reward = reward + cumulate_reward * 0.99

            if(step > 200) and (step % 5 == 0):
                model.learn(store_cost_flag)
                store_cost_flag = False

            counter += 1
            # break while loop when end of this episode
            if counter > 300 or done:
                break

            env_s = env_s_
            step += 1

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




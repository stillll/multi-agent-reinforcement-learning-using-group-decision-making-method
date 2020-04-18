import tensorflow as tf
import os
import pdb
import numpy as np

def train_model(env, model, save_path, max_episode):
    step = 0
    cumulate_reward = 0
    accident = False
    for episode in range(max_episode):
        if episode % 1000 == 0:
            print("train_episode", episode)
        env_s = env.reset()  # init env and return env state
        store_cost_flag = True  # if store cost
        counter = 0  # if end episode
        #cumulate_reward = 0
        join_act, w_r, sugg_act = model.run_model(env_s)  # 第一步
        while True:  # one step

            # learn
            step += 1
            # break while loop when end of this episode

            #print("last",last_join_act)
            w_r_ = w_r  # 上一步的奖励系数
            last_env_s = env_s
            try:
                env_s, reward, done = env.step(join_act)  # 当前步
            except:
                accident = True
                break
            cumulate_reward = reward + cumulate_reward
            last_join_act = join_act  # 上一步的动作
            last_sugg_act = sugg_act
            obv = model.n_obv  # 上一步的观察
            join_act, w_r, sugg_act = model.run_model(env_s)  # 当前步
            #print("now",join_act)
            model.store_n_transitions(obv, last_join_act, last_sugg_act, reward, w_r_)  # 上一步到当前步的转移经验
            if counter > 300 or done:
                break
            counter += 1
            if step % 5 == 0:
                model.learn(True)
                store_cost_flag = False


        # record cumulate rewards once an episode
        if episode % 1000 == 0:
            print("reward:", cumulate_reward)
        model.reward_his.append(cumulate_reward)
        if accident:
            break

    # save model
    saver = tf.train.Saver()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saver.save(model.sess, save_path)
    # end of game
    print('game over')
    #if accident is False:
        #env.destroy()

    if not os.path.exists('data_for_plot'):
        os.makedirs('data_for_plot')
    write_rewards = open('data_for_plot/'+str(model.n_agents)+'-'+str(model.max_coop)+'-reward_his.txt', 'w+')
    for ip in model.reward_his: 
        write_rewards.write(str(ip))
        write_rewards.write('\n')
    write_rewards.close()

    write_costs = open('data_for_plot/'+str(model.n_agents)+'-'+str(model.max_coop)+'-cost_his.txt', 'w+')
    for ip in model.reward_his:
        write_costs.write(str(ip))
        write_costs.write('\n')
    write_costs.close()

    model.plot_cost()
    model.plot_reward()


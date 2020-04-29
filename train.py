import tensorflow as tf
import os
import pdb
import numpy as np


def train_model(env, model, save_path, max_episode):
    step = 0
    cumulate_reward = 0
    accident = False
    for episode in range(max_episode):
        #if episode % 1000 == 1:
            #model.plot_cost()
            #model.plot_reward()
            #model.plot_actions_value()
            #for i in range(9):
                #aaa = []
                #for j in range(9):
                    #aaa.append(np.max(model.value_func(np.array([-5+j,-5+i,-20,-20]))))
                #print(aaa)
            #pdb.set_trace()
            #print(np.max(model.value_func(np.array([-5.,-5.]))),model.value_func([-4.,-5.]),model.value_func([-3.,-5.]),model.value_func([-2.,-5.]),model.value_func([-1.,-5.]),model.value_func([0.,-5.]),model.value_func([1.,-5.]),model.value_func([2.,-5.]),model.value_func([3.,-5.]))
            #print(model.value_func([-5,-4]),model.value_func([-4,-4]),model.value_func([-3,-4]),model.value_func([-2,-4]),model.value_func([-1,-4]),model.value_func([0,-4]),model.value_func([1,-4]),model.value_func([2,-4]),model.value_func([3,-4]))
            #print(model.value_func([-5,-3]),model.value_func([-4,-3]),model.value_func([-3,-3]),model.value_func([-2,-3]),model.value_func([-1,-3]),model.value_func([0,-3]),model.value_func([1,-3]),model.value_func([2,-3]),model.value_func([3,-3]))
            #rint(model.value_func([-5,-2]),model.value_func([-4,-2]),model.value_func([-3,-2]),model.value_func([-2,-2]),model.value_func([-1,-2]),model.value_func([0,-2]),model.value_func([1,-2]),model.value_func([2,-2]),model.value_func([3,-2]))
            #rint(model.value_func([-5,-1]),model.value_func([-4,-1]),model.value_func([-3,-1]),model.value_func([-2,-1]),model.value_func([-1,-1]),model.value_func([0,-1]),model.value_func([1,-1]),model.value_func([2,-1]),model.value_func([3,-1]))
            #print(model.value_func([-5,0]),model.value_func([-4,0]),model.value_func([-3,0]),model.value_func([-2,0]),model.value_func([-1,0]),model.value_func([0,0]),model.value_func([1,0]),model.value_func([2,0]),model.value_func([3,0]))
            #print(model.value_func([-5,1]),model.value_func([-4,1]),model.value_func([-3,1]),model.value_func([-2,1]),model.value_func([-1,1]),model.value_func([0,1]),model.value_func([1,1]),model.value_func([2,1]),model.value_func([3,1]))
            #print(model.value_func([-5,2]),model.value_func([-4,2]),model.value_func([-3,2]),model.value_func([-2,2]),model.value_func([-1,2]),model.value_func([0,2]),model.value_func([1,2]),model.value_func([2,2]),model.value_func([3,2]))
            #print(model.value_func([-5,3]),model.value_func([-4,3]),model.value_func([-3,3]),model.value_func([-2,3]),model.value_func([-1,3]),model.value_func([0,3]),model.value_func([1,3]),model.value_func([2,3]),model.value_func([3,3]))
        if episode % 10 == 0:
            print("train_episode", episode)
        env_s = env.reset()  # init env and return env state
        store_cost_flag = True  # if store cost
        counter = 0  # if end episode
        #cumulate_reward = 0
        join_act, w_r, sugg_act = model.run_model(env_s)  # 第一步
        # 1、DQN（x,y,z,...）
        # 2、GDM（a,b,c,...）
        while True:  # one step
            # learn
            step += 1
            # break while loop when end of this episode

            #print("last",last_join_act)
            w_r_ = w_r  # 上一步的奖励系数
            last_env_s = env_s
            try:
                env_s, reward, done = env.step(join_act)  # 当前步
                env.render()
            except:
                accident = True
                break
            cumulate_reward = reward + cumulate_reward
            last_join_act = join_act  # 上一步的动作
            last_sugg_act = sugg_act
            obv = model.n_obv  # 上一步的观察
            join_act, w_r, sugg_act = model.run_model(env_s)  # 当前步
            # for agents
            #   coop set; -- son of dqn
            #   DQN; -- once
            #   GDM; -- son of coop set
            #print("now",join_act)
            model.store_n_transitions(obv, last_join_act, last_sugg_act, reward, w_r_)  # 上一步到当前步的转移经验
            if counter > 300 or done:
                break
            counter += 1
            if step % 5 == 0:
                model.learn(True)
                store_cost_flag = False


        # record cumulate rewards once an episode
        if episode % 10 == 0:
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
    model.plot_actions_value()


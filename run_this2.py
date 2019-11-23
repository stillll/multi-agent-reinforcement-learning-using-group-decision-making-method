from maze_env2 import Maze
from RL_brain import DeepQNetwork
import numpy as np
import itertools as itr


def coop_set_and_coop_state(agent_id, env_s, pow_set):
    max_value = -10
    for each in pow_set:  # 'each' is an array, subsets with different lengths
        for each_ in each:  # 'each_' is a array, a specific item in a subset which has specific length
            if each_[0] == agent_id:
                actual_coop = len(each_)
                tmp_value = 0
                y = [a * 2 for a in each_]
                x = [a + 1 for a in y]
                r = sorted(y + x)
                coop_state = env_s[r]
                if len(coop_state) < env.n_features*env.max_coop:
                    actual_len = len(coop_state)
                    coop_state = np.append(coop_state, [-999]*(env.max_coop*env.n_features-actual_len))
                q_values = RL.value_func(coop_state)
                for one in q_values[0, :actual_coop*env.n_actions]:
                    tmp_value += one
                tmp_value /= actual_coop
                if tmp_value > max_value:
                    max_value = tmp_value
                    coop_state_i = coop_state
                    coop_set = each_
    return coop_state_i, coop_set


def run_maze():
    step = 0
    all_agent_id = np.array(range(env.n_agents))
    pow_set = []
    for i in range(1, env.max_coop+1):
        pow_set.append(np.array(list(itr.permutations(all_agent_id, i))))
    reward_his = []
    cumulate_reward = 0
    for episode in range(100000):
        # initial observation
        print("episode", episode, '\n')
        env.reset()  # init env
        env_s_ = env.env_s()
        flag = True
        counter = 0
        while True:
            # RL choose action based on observation and suggestion for every agents
            env_s = env_s_
            join_act = []
            n_obv = []
            for i in range(env.n_agents):
                """
                coop_set = coop_set(i)
                ob = obv(i,coop_set)
                Q = Q(ob)
                suggestion = GDM(Q)
                choose_action(i, suggestion)
                """
                observation, coop_set = coop_set_and_coop_state(i, env_s, pow_set)
                n_obv.append(observation)
                action = RL.choose_action(observation)
                join_act.append(action)

            # RL take action and get next observation and reward
            reward, done = env.step(join_act)
            cumulate_reward += reward
            counter += 1
            if counter > 300:
                break

            # fresh env
            env.render()
            env_s_ = env.env_s()
            for i in range(env.n_agents):
                observation_ = coop_set_and_coop_state(i, env_s_, pow_set)[0]
                coop_act = np.array(join_act)[coop_set]
                if len(coop_act) < env.max_coop:
                    coop_act = np.append(coop_act, [-1]*(env.max_coop-len(coop_act)))
                RL.store_transition(n_obv[i], coop_act, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn(flag)
                flag = False

            # break while loop when end of this episode
            if done:
                break
            step += 1
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

if __name__ == "__main__":
    # maze game
    env = Maze()

    RL = DeepQNetwork(max_coop=env.max_coop,
                      n_agents=env.n_agents,
                      n_actions=env.n_actions,
                      n_features=env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()

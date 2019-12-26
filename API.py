import argparse
# from WAgent.python import magent
from maze_env2 import Maze
from Model import MAGDMRL
import tensorflow as tf
from test import test_model
from train import train_model
import numpy as np
import os
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="pursuit", help="name of the scenario script")
    parser.add_argument("--map_size", type=int, default=100, help="the size of map")
    parser.add_argument("--num_1", type=int, default=3, help="the number of group_1")
    parser.add_argument("--num_2", type=int, default=20, help="the number of group_2")
    parser.add_argument("--num_walls", type=int, default=100, help="the number of the walls")

    # model attributes
    parser.add_argument("--alg", default='dqn', choices=['dqn', 'drqn', 'a2c', 'gdm'])
    parser.add_argument("--max_step", type=int, default=300, help='the max step of a episode')
    parser.add_argument("--save_path", type=str, default="3-2gdm-discuss/", help="the path to save")
    parser.add_argument("--model_name", type=str, default="MAGDMRL", help="the path for model to save")
    parser.add_argument("--model_exist", type=bool, default=False, help="if use the exist model")
    parser.add_argument("--num_episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--max_coop", type=int, default=2, help="max number of cooperate agents")
    parser.add_argument("--use_gdm", type=bool, default=True, help="if use the gdm policy")
    parser.add_argument("--n_features", type=int, default=2, help="the feature dimension")

    parser.add_argument("--max_discuss", type=int, default=1)

    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.90, help="discount factor")
    parser.add_argument("--e_greedy", type=float, default=0.90, help="greedy degree")
    parser.add_argument("--batch_size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--cll_ba", type=float, default=0.50, help="threshold of cooperate level")
    parser.add_argument("--replace_target_iter", type=int, default=500, help="update rate for target network")
    parser.add_argument("--memory_size", type=int, default=2000, help="the size of memory pool")
    # save and display
    parser.add_argument("--output_graph", type=bool, default=False, help="if display the graph")

    # others
    parser.add_argument("--e_greedy_add", type=bool, default=None, help="e_greedy_increment")
    parser.add_argument("--benchmark_dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots_dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


'''
def make_env(arglist):
    env = magent.GridWorld(arglist.scenario, map_size=arglist.map_size)
    env.set_render_dir("build/render")
    # get group handles
    group_1, group_2 = env.get_handles()
    # init env and agents
    env.reset()
    env.add_walls(method="random", n=arglist.num_walls)
    env.add_agents(group_1, method="random", n=arglist.num_1)
    env.add_agents(group_2, method="random", n=arglist.num_2)
    return env
'''


def make_env(arglist):
    env = Maze(n_agents=arglist.num_1,
               max_coop=arglist.max_coop)
    return env


def set_model(arglist, env):

    model = MAGDMRL(n_agents=arglist.num_1,
                    n_actions=4,
                    max_coop=arglist.max_coop,
                    n_features=arglist.n_features,  # feature length for single agent
                    cll_ba=arglist.cll_ba,
                    # max_discuss=arglist.max_discuss,
                    learning_rate=arglist.lr,
                    reward_decay=arglist.gamma,
                    e_greedy=arglist.e_greedy,
                    replace_target_iter=arglist.replace_target_iter,
                    memory_size=arglist.memory_size,
                    batch_size=arglist.batch_size,
                    e_greedy_increment=arglist.e_greedy_add,
                    output_graph=arglist.output_graph,
                    use_gdm=arglist.use_gdm,
                    sess=None)
    return model


def train_or_test(arglist):
    env = make_env(arglist)
    model = set_model(arglist, env)
    '''
    group_1, group_2 = env.get_handles()
    done = False
    step = 0
    while not done:
        print("nums: %d vs %d" % (env.get_num(group_1), env.get_num(group_2)))
        num_1 = env.get_num(group_1)  # int type
        alive_1 = env.get_alive(group_1)  # bool type, shape: [True, ... , True]
        ids_1 = env.get_agent_id(group_1)  # shape: [1, ... , n]
        pos_1 = env.get_pos(group_1)  # shape: [[x1, y1], ... , [xn, yn]]
        obs_1 = model.cg(num_1, alive_1, ids_1, pos_1)
        acts_1 = model.take_action(obs_1, ids_1)
        env.set_action(group_1, acts_1)
        reward_1 = env.get_reward(group_1)  # shape: [1 * n]

        num_2 = env.get_num(group_2)
        alive_2 = env.get_alive(group_2)
        ids_2 = env.get_agent_id(group_2)
        pos_2 = env.get_pos(group_2)
        obs_2 = model.cg(num_2, alive_2, ids_2, pos_2)
        acts_2 = model.take_action(obs_2, ids_2)
        env.set_action(group_2, acts_2)
        reward_2 = env.get_reward(group_2)

        # simulate one step
        done = env.step()

        # render
        env.render()

        # get reward
        reward = [sum(reward_1), sum(reward_2)]

        # clear dead agents
        env.clear_dead()

        # print info
        if step % 10 == 0:
            print("step: %d\t predators' reward: %d\t preys' reward: %d" %
                  (step, reward[0], reward[1]))

        step += 1
        if step > arglist.max_step:
            break
    '''
    if arglist.model_exist:
        saver = tf.train.Saver()
        saver.restore(model.sess, tf.train.latest_checkpoint(arglist.save_path))
        env.after(100, test_model(env, model, max_episode=arglist.num_episodes))
        env.mainloop()
    else:
        env.after(100, train_model(env, model, arglist.save_path + arglist.model_name, max_episode=arglist.num_episodes))
        env.mainloop()


if __name__ == "__main__":
    arglist = parse_args()
    train_or_test(arglist)

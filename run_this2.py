import argparse
from maze_env2 import Maze
import numpy as np
import tensorflow as tf
from Model import MAGDMRL
from train import train_model
from test import test_model

np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="train_pursuit", help="name of the scenario script")

    parser.add_argument("--max_step", type=int, default=300, help="maximum episode length")
    parser.add_argument("--map_size", type=int, default=1000, help="the size of map")
    parser.add_argument("--num_episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num_agents", type=int, default=3, help="number of agents")
    parser.add_argument("--max_coop", type=int, default=3, help="max number of cooperate agents")
    parser.add_argument("--use_GDM", type=bool, default=False, help="if use the gdm policy")
    parser.add_argument("--n_features", type=int, default=2, help="the feature dimension")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.90, help="discount factor")
    parser.add_argument("--e_greedy", type=float, default=0.90, help="greedy degree")
    parser.add_argument("--batch_size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--cll_ba", type=float, default=0.50, help="threshold of cooperate level")
    parser.add_argument("--replace_target_iter", type=int, default=500, help="update rate for target network")
    parser.add_argument("--memory_size", type=int, default=2000, help="the size of memory pool")
    # save and display
    parser.add_argument("--output_graph", type=bool, default=False, help="if display the graph")
    parser.add_argument("--model_name", type=str, default="MAGDMRL", help="MUST GIVE THE NAME OF THE MODEL!!!")
    parser.add_argument("--save_dir", type=str, default="3-2model/", help="directory in which training state and model should be saved")
    parser.add_argument("--load_path", type=str, default="3-2model/", help="directory in which training state and model are loaded")
    # others
    parser.add_argument("--e_greedy_add", type=bool, default=None, help="e_greedy_increment")
    parser.add_argument("--benchmark_dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots_dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


def load_model(arglist):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(arglist.load_path + arglist.model_name + '.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(arglist.load_path))
    model = MAGDMRL(n_agents=arglist.num_agents,
                    n_actions=4,
                    max_coop=arglist.max_coop,
                    n_features=arglist.n_features,  # feature length for single agent
                    cll_ba=arglist.cll_ba,
                    learning_rate=arglist.lr,
                    reward_decay=arglist.gamma,
                    e_greedy=arglist.e_greedy,
                    replace_target_iter=arglist.replace_target_iter,
                    memory_size=arglist.memory_size,
                    batch_size=arglist.batch_size,
                    e_greedy_increment=arglist.e_greedy_add,
                    output_graph=arglist.output_graph,
                    use_gdm=arglist.use_GDM,
                    sess=sess)
    return model


def make_env(arglist):
    from WAgent.python import magent
    # load scenario from script
    #env = config.load(scenario_name + ".py")
    env = magent.GridWorld(arglist.scenario, arglist.map_size)
    return env



def train(arglist):
    env = make_env(arglist)
    model = load_model(arglist)
    env.after(100, test_model(env, model, max_episode=100))
    env.mainloop()


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

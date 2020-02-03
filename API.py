import argparse
from maze import Maze
#from coop_env import Coop
from Model import MAGDMRL
import tensorflow as tf
from test import test_model
from train import train_model


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--MAgent", action='store_true', help="if use the MAgent Env")
    parser.add_argument("--scenario", type=str, default="pursuit", help="name of the scenario script")
    parser.add_argument("--map_size", type=int, default=30, help="the size of map")
    parser.add_argument("--num_agent", type=int, default=3, help="the number of agents")
    parser.add_argument("--num_goal", type=int, default=3, help="the number of targets")
    parser.add_argument("--num_walls", type=int, default=100, help="the number of the walls")

    # model attributes
    parser.add_argument("--alg", default='dqn', choices=['dqn', 'drqn', 'a2c', 'gdm'])
    parser.add_argument("--max_step", type=int, default=300, help='the max step of a episode')
    parser.add_argument("--save_path", type=str, default="3-2gdm-discuss/", help="the path to save")
    parser.add_argument("--model_name", type=str, default="MAGDMRL", help="the path for model to save")
    parser.add_argument("--model_exist", action="store_true", help="if use the exist model")
    parser.add_argument("--max_episode", type=int, default=1000, help="the number of episodes")
    parser.add_argument("--max_coop", type=int, default=2, help="max number of cooperate agents")
    parser.add_argument("--gdm", action="store_true", help="if use the gdm policy")
    parser.add_argument("--n_features", type=int, default=2, help="the feature dimension")
    parser.add_argument("--max_discuss", type=int, default=5, help="the number of discussions")
    parser.add_argument("--act_space", type=int, default=4, help="the action space,4 or 13")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.90, help="discount factor")
    parser.add_argument("--e_greedy", type=float, default=0.90, help="greedy degree")
    parser.add_argument("--batch_size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--cll_ba", type=float, default=0.50, help="threshold of cooperate level")
    parser.add_argument("--replace_target_iter", type=int, default=500, help="update rate for target network")
    parser.add_argument("--memory_size", type=int, default=2000, help="the size of memory pool")
    # save and display
    parser.add_argument("--output_graph", action="store_true", help="if display the graph")

    # others
    parser.add_argument("--e_greedy_add", type=bool, default=None, help="e_greedy_increment")
    parser.add_argument("--benchmark_dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots_dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def make_env(arglist):
    #if arglist.MAgent:
        #env = Coop(arglist)
    #else:
    env = Maze(n_agents=arglist.num_agent)
    return env


def set_model(arglist):
    model = MAGDMRL(n_agents=arglist.num_agent,
                    n_actions=arglist.act_space,
                    max_coop=arglist.max_coop,
                    n_features=arglist.n_features,  # feature length for single agent
                    cll_ba=arglist.cll_ba,
                    max_discuss=arglist.max_discuss,
                    learning_rate=arglist.lr,
                    reward_decay=arglist.gamma,
                    e_greedy=arglist.e_greedy,
                    replace_target_iter=arglist.replace_target_iter,
                    memory_size=arglist.memory_size,
                    batch_size=arglist.batch_size,
                    e_greedy_increment=arglist.e_greedy_add,
                    output_graph=arglist.output_graph,
                    use_gdm=arglist.gdm,
                    sess=None)
    return model


def train_or_test(arglist):
    env = make_env(arglist)
    model = set_model(arglist)

    if arglist.model_exist:
        saver = tf.train.Saver()
        saver.restore(model.sess, tf.train.latest_checkpoint(arglist.save_path))
        test_model(env, model, max_episode=100)
    else:
        train_model(env, model, arglist.save_path + arglist.model_name, max_episode=arglist.max_episode)


if __name__ == '__main__':
    arglist = parse_args()
    train_or_test(arglist)


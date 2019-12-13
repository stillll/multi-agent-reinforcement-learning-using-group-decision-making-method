from maze_env2 import Maze
import numpy as np
import tensorflow as tf
from Model import MAGDMRL
from train import train_model
from test import test_model
np.random.seed(0)


def load_model(load_path, model_name):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(load_path+model_name+'.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(load_path))
    model = MAGDMRL(n_agents=3,
                    n_actions=4,
                    max_coop=3,
                    n_features=2,  # feature length for single agent
                    cll_ba=0.5,
                    learning_rate=0.01,
                    reward_decay=0.9,
                    e_greedy=0.9,
                    replace_target_iter=500,
                    memory_size=2000,
                    batch_size=32,
                    e_greedy_increment=None,
                    output_graph=False,
                    use_gdm=False,
                    sess=sess)
    return model


if __name__ == "__main__":
    save_path = "3-2model/"
    model_name = "MAGDMRL"
    save_path2 = "3-1model_non_gdm/"
    model_name2 = "MARL"

    model_exist = True

    # maze game
    env = Maze(n_agents=3,
               max_coop=2)

    if model_exist is True:
        model = load_model(save_path, model_name)
        env.after(100, test_model(env, model, max_episode=100))
        env.mainloop()
    else:
        model = MAGDMRL(n_agents=3,
                        n_actions=4,
                        max_coop=2,
                        n_features=2,  # feature length for single agent
                        cll_ba=0.5,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=500,
                        memory_size=2000,
                        batch_size=32,
                        e_greedy_increment=None,
                        output_graph=True,
                        use_gdm=True,
                        sess=None)
        env.after(100, train_model(env, model, save_path+model_name, max_episode=1000))
        env.mainloop()




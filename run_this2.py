from maze_env2 import Maze
import numpy as np
import tensorflow as tf
from Model import MAGDMRL
from train import train_model
from test import test_model
np.random.seed(0)


if __name__ == "__main__":
    save_path = "3-2gdm-discuss/"
    model_name = "MAGDMRL"
    save_path2 = "3-2model_non_gdm/"
    model_name2 = "MARL"

    model_exist = False

    # maze game
    env = Maze(n_agents=3,
               max_coop=2)

    if model_exist is True:
        model = MAGDMRL(n_agents=3,
                        n_actions=4,
                        max_coop=2,
                        n_features=2,  # feature length for single agent
                        cll_ba=0.5,
                        max_discuss=1,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=500,
                        memory_size=2000,
                        batch_size=32,
                        e_greedy_increment=None,
                        output_graph=False,
                        use_gdm=True,
                        sess=None)
        saver = tf.train.Saver()
        saver.restore(model.sess, tf.train.latest_checkpoint(save_path))
        env.after(100, test_model(env, model, max_episode=100))
        env.mainloop()
    else:
        model = MAGDMRL(n_agents=3,
                        n_actions=4,
                        max_coop=2,
                        n_features=2,  # feature length for single agent
                        cll_ba=0.5,
                        max_discuss=1,
                        learning_rate=0.01,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=400,
                        memory_size=2000,
                        batch_size=32,
                        e_greedy_increment=None,
                        output_graph=False,
                        use_gdm=True,
                        sess=None)
        env.after(100, train_model(env, model, save_path+model_name, max_episode=1000))
        env.mainloop()

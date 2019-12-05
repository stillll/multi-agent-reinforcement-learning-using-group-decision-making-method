from maze_env2 import Maze
from RL_brain import DeepQNetwork
import numpy as np
from GDM import GroupDM
from Model import run_model
np.random.seed(0)

if __name__ == "__main__":
    # maze game
    env = Maze(n_agents=3,
               max_coop=2)
    dqn = DeepQNetwork(max_coop=env.max_coop,
                       n_agents=env.n_agents,
                       n_actions=env.n_actions,
                       n_features=env.n_features,
                       learning_rate=0.01,
                       reward_decay=0.9,
                       e_greedy=0.9,
                       replace_target_iter=300,
                       memory_size=2000,
                       batch_size=64,
                       e_greedy_increment=None,
                       output_graph=False)
    gdm = GroupDM(n_actions=env.n_actions,
                  n_agents=env.n_agents,
                  max_coop=env.max_coop,
                  cll_ba=0.5)
    env.after(100, run_model(env, dqn, gdm))
    env.mainloop()
    dqn.plot_cost()

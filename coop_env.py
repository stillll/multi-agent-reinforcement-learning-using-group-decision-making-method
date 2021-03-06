import numpy as np
import sys
#from MAS_Env.MAS_Catch import GameEnv
import matplotlib.pyplot as plt
from MAS_Env.MAS_Switch import GameEnv
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import pdb

class Coop(tk.Tk):
    def __init__(self, args):
        self.scenario = args.scenario
        self.map_size = args.map_size
        self.num_agent = args.num_agent
        self.num_goal = args.num_goal
        self.num_walls = args.num_walls
        self.n_actions = 7
        self.env = self._build_env()

    def _build_env(self):
        # if self.scenario == "MAS_Catch":
        #     from MAS_Env.MAS_Catch import GameEnv
        # elif self.scenario == "MAS_Switch":
        #     from MAS_Env.MAS_Switch import GameEnv

        env = GameEnv(
            width=self.map_size,
            high=self.map_size,
            agent_num=self.num_agent,
            food_num=self.num_goal,
            agent_blood=1,
            food_blood=1
        )
        return env

    def reset(self):
        self.env.reset()
        pos_agents = []
        pos_foods = []

        for i in range(self.num_agent):
            pos_agents.append([self.env.agent_objects[i].type, self.env.agent_objects[i].x, self.env.agent_objects[i].y])
        for j in range(self.num_goal):
            pos_foods.append([self.env.food_objects[j].type, self.env.food_objects[j].x, self.env.food_objects[j].y])

        pos_observation = pos_agents + pos_foods
        pos_observation = np.array(pos_observation)
        # return observation
        return pos_observation.flatten()

    def step(self, action):
        action = np.array(action)
        # reward function
        reward_list, done = self.env.move(action)
        reward = np.sum(reward_list)
        pos_agents = []
        pos_foods = []

        for i in range(self.num_agent):
            pos_agents.append([self.env.agent_objects[i].type, self.env.agent_objects[i].x, self.env.agent_objects[i].y])
        for j in range(self.num_goal):
            pos_foods.append([self.env.food_objects[j].type, self.env.food_objects[j].x, self.env.food_objects[j].y])

        pos_observation = pos_agents + pos_foods
        pos_observation = np.array(pos_observation)
        # return observation

        return pos_observation.flatten(), reward, done

    def render(self):
        # temp = self.env.render_env()
        temp = self.env.train_render()
        plt.imshow(temp)
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()





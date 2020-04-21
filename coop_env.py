import numpy as np
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
from MAS_Gathering import GameEnv
import matplotlib.pyplot as plt


class Coop(tk.Tk):
    def __init__(self, args):
        self.scenario = args.scenario
        self.map_size = args.map_size
        self.num_agent = args.num_agent
        self.num_goal = args.num_goal
        self.num_walls = args.num_walls
        self.n_actions = 8
        self.env = self._build_env()

    def _build_env(self):
        env = GameEnv()
        # env.reset()
        return env

    def reset(self):
        pos_1 = [self.env.agent1.x - self.env.food_objects[0].x, self.env.agent1.y - self.env.food_objects[0].y]
        pos_2 = [self.env.agent2.x - self.env.food_objects[0].x, self.env.agent2.y - self.env.food_objects[0].y]
        pos_agents = [pos_1, pos_2]  # shape: [[x1, y1], ... , [xn, yn]]
        pos_agents = np.array(pos_agents)

        # return observation
        return pos_agents.flatten()

    def step(self, action):
        action = np.array(action)

        # reward function
        reward_1, reward_2, done = self.env.move(action[0], action[1])
        reward = reward_1 + reward_2

        pos_1 = [self.env.agent1.x - self.env.food_objects[0].x, self.env.agent1.y - self.env.food_objects[0].y]
        pos_2 = [self.env.agent2.x - self.env.food_objects[0].x, self.env.agent2.y - self.env.food_objects[0].y]
        pos_agents = [pos_1, pos_2]  # shape: [[x1, y1], ... , [xn, yn]]
        pos_agents = np.array(pos_agents)

        return pos_agents.flatten(), reward, done

    def render(self):
        temp = self.env.render_env()
        plt.imshow(temp)
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()





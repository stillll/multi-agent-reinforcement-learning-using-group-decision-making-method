import numpy as np
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import magent


class Coop(tk.Tk):
    def __init__(self, args):
        self.scenario = args.scenario
        self.map_size = args.map_size
        self.num_agent = args.num_agent
        self.num_goal = args.num_goal
        self.num_walls = args.num_walls
        self.env = self._build_env()

    def _build_env(self):
        magent.utility.init_logger(self.scenario)
        env = magent.GridWorld(self.scenario, map_size=self.map_size)
        # env = super(Coop, self).__init__(self.scenario, map_size=self.map_size)
        # env.set_render_dir("build/render")
        env.reset()
        env.add_walls(method="random", n=self.num_walls)
        env.add_agents(env.get_handles()[0], method="random", n=self.num_agent)
        env.add_agents(env.get_handles()[1], method="random", n=self.num_goal)
        return env

    def reset(self):
        num_agents = self.env.get_num(self.env.get_handles()[0])  # int type
        alive_agents = self.env.get_alive(self.env.get_handles()[0])  # bool type, shape: [True, ... , True]
        ids_agents = self.env.get_agent_id(self.env.get_handles()[0])  # shape: [1, ... , n]
        pos_agents = self.env.get_pos(self.env.get_handles()[0])  # shape: [[x1, y1], ... , [xn, yn]]
        pos_agents = np.array(pos_agents)
        '''
        num_goals = self.env.get_num(self.env.get_handles()[1])  # int type
        alive_goals = self.env.get_alive(self.env.get_handles()[1])  # bool type, shape: [True, ... , True]
        ids_goals = self.env.get_agent_id(self.env.get_handles()[1])  # shape: [1, ... , n]
        pos_goals = self.env.get_pos(self.env.get_handles()[1])  # shape: [[x1, y1], ... , [xn, yn]]
        pos_goals = np.array(pos_goals)
        '''
        # return observation
        return pos_agents.flatten()

    def step(self, action):
        action = np.array(action)
        action = action.astype(np.int32)
        self.env.set_action(self.env.get_handles()[0], action)
        self.env.clear_dead()

        # if game over
        done = self.env.step()

        pos_agents = self.env.get_pos(self.env.get_handles()[0])
        pos_agents = np.array(pos_agents)

        # reward function
        rewards = self.env.get_reward(self.env.get_handles()[0])
        reward = sum(rewards)

        return pos_agents.flatten(), reward, done

    def render(self):
        self.env.render()





"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
import random
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 9  # grid height
MAZE_W = 9  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.n_agents = 3
        self.max_coop = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self,):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        loc_c = random.randint(2, 4)
        loc_r = random.randint(2, 4)
        hell1_center = origin + np.array([UNIT * loc_c, UNIT * loc_r])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # create oval
        loc = random.randint(5, 8)
        oval_center = origin + UNIT * loc
        if (oval_center.all == hell1_center).all():
            oval_center += UNIT
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # create green rect
        self.rect2 = self.canvas.create_rectangle(
            origin[0] + 320 - 15, origin[1] - 15,
            origin[0] + 320 + 15, origin[1] + 15,
            fill='green')

        # create blue rect
        self.rect3 = self.canvas.create_rectangle(
            origin[0] + 320 - 15, origin[1] + 320 - 15,
            origin[0] + 320 + 15, origin[1] + 320 + 15,
            fill='blue')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.oval)
        self.canvas.delete(self.rect)
        self.canvas.delete(self.rect2)
        self.canvas.delete(self.rect3)
        origin = np.array([20, 20])
        #loc = random.randint(5, 8)
        oval_center = origin + UNIT * 5
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        self.rect2 = self.canvas.create_rectangle(
            origin[0] + 320 - 15, origin[1] - 15,
            origin[0] + 320 + 15, origin[1] + 15,
            fill='green')
        self.rect3 = self.canvas.create_rectangle(
            origin[0] + 320 - 15, origin[1] + 320 - 15,
            origin[0] + 320 + 15, origin[1] + 320 + 15,
            fill='blue')
        # return observation
        """
        return np.hstack(((np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT),
                         (np.array(self.canvas.coords(self.rect2)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT),
                         (np.array(self.canvas.coords(self.rect3)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)))
        """

    def env_s(self, ):
        return np.hstack(
                        ((np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT),
                         (np.array(self.canvas.coords(self.rect2)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT),
                         (np.array(self.canvas.coords(self.rect3)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
                        )
                        )

    def step(self, action):
        """
        oval_s = self.canvas.coords(self.oval)
        hell_s = self.canvas.coords(self.hell1)
        oval_act = random.randint(0, 3)
        base_act = np.array([0, 0])
        if oval_act == 0:  # up
            if oval_s[1] > UNIT:
                base_act[1] -= UNIT
        elif oval_act == 1:  # down
            if oval_s[1] < (MAZE_H - 1) * UNIT:
                base_act[1] += UNIT
        elif oval_act == 2:  # right
            if oval_s[0] < (MAZE_W - 1) * UNIT:
                base_act[0] += UNIT
        elif oval_act == 3:  # left
            if oval_s[0] > UNIT:
                base_act[0] -= UNIT
        if oval_s[0]+base_act[0] != hell_s[0] and oval_s[1]+base_act[1] != hell_s[1]:
            self.canvas.move(self.oval, base_act[0], base_act[1])
        """

        for i in range(self.n_agents):
            if i == 0:
                s = self.canvas.coords(self.rect)
            elif i == 1:
                s = self.canvas.coords(self.rect2)
            elif i == 2:
                s = self.canvas.coords(self.rect3)
            base_action = np.array([0, 0])
            if action[i] == 0:   # up
                if s[1] > UNIT:
                    base_action[1] -= UNIT
            elif action[i] == 1:   # down
                if s[1] < (MAZE_H - 1) * UNIT:
                    base_action[1] += UNIT
            elif action[i] == 2:   # right
                if s[0] < (MAZE_W - 1) * UNIT:
                    base_action[0] += UNIT
            elif action[i] == 3:   # left
                if s[0] > UNIT:
                    base_action[0] -= UNIT
            if i == 0:
                self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
            elif i == 1:
                self.canvas.move(self.rect2, base_action[0], base_action[1])
            elif i == 2:
                self.canvas.move(self.rect3, base_action[0], base_action[1])

        next_coords = self.canvas.coords(self.rect)  # next state
        next_coords2 = self.canvas.coords(self.rect2)
        next_coords3 = self.canvas.coords(self.rect3)

        # reward function
        if next_coords == self.canvas.coords(self.oval) or next_coords2 == self.canvas.coords(self.oval) \
                or next_coords3 == self.canvas.coords(self.oval):
            reward = 5
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)] or next_coords2 in [self.canvas.coords(self.hell1)]\
                or next_coords3 in [self.canvas.coords(self.hell1)]:
            reward = -2
            done = False
        elif next_coords == next_coords2 or next_coords2 == next_coords3 or next_coords3 == next_coords:
            reward = -1
            done = False
        else:
            reward = -0.1
            done = False
        #s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        #s_2 = (np.array(next_coords2[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)
        #s_3 = (np.array(next_coords3[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)
        return reward, done

    def render(self):
        time.sleep(0.01)
        self.update()



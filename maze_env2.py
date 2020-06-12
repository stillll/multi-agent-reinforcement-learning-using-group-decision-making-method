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

import pdb


class Maze(tk.Tk, object):
    def __init__(self,
                 n_agents=3,
                 max_coop=2,
                 unit = 40,
                 maze_h = 9,
                 maze_w = 9,
                 rect_size = 32,
                 rect_pos = [],
                 show = True
                 ):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.n_agents = n_agents
        self.max_coop = max_coop
        self.maze_h = maze_h
        self.maze_w = maze_w
        self.unit = unit
        self.rect_size = rect_size
        self.rect_pos_list = rect_pos
        self.rect_pos_list_his = rect_pos.copy()
        self.rect_list = []
        self.last_dis = 0
        self.done = False
        self.show = show
        if self.show:
            self.title('maze')
            self.geometry('{0}x{1}'.format(self.maze_w * self.unit, self.maze_h * self.unit))
        self._build_maze()
        

    def _build_maze(self,):
        hell_loc_h = random.randint(0, self.maze_h-1)
        hell_loc_w = random.randint(0, self.maze_w-1)
        oval_loc_h = random.randint(0, self.maze_h-1)
        oval_loc_w = random.randint(0, self.maze_w-1)
        if (hell_loc_h == oval_loc_h and hell_loc_w == oval_loc_w):
            oval_loc_h += 1
        if oval_loc_h > self.maze_h - 1:
            oval_loc_h = oval_loc_h - self.maze_h
        for i in range(self.n_agents):
            if i*self.n_features >= len(self.rect_pos_list):
                loc_h = random.randint(0, self.maze_h-1)
                loc_w = random.randint(0, self.maze_w-1)
                self.rect_pos_list.append(loc_w)
                self.rect_pos_list.append(loc_h)
        if self.show:
            self.canvas = tk.Canvas(self, bg='white',
                            height=self.maze_h * self.unit,
                            width=self.maze_w * self.unit)

            # create grids
            for c in range(0, self.maze_w * self.unit, self.unit):
                x0, y0, x1, y1 = c, 0, c, self.maze_h * self.unit
                self.canvas.create_line(x0, y0, x1, y1)
            for r in range(0, self.maze_h * self.unit, self.unit):
                x0, y0, x1, y1 = 0, r, self.maze_w * self.unit, r
                self.canvas.create_line(x0, y0, x1, y1)

            # create origin
            origin = np.array([self.unit/2, self.unit/2])

            # hell
            hell1_center = origin + np.array([self.unit * hell_loc_w, self.unit * hell_loc_h])
            self.hell1 = self.canvas.create_rectangle(
                hell1_center[0] - self.rect_size/2, hell1_center[1] - self.rect_size/2,
                hell1_center[0] + self.rect_size/2, hell1_center[1] + self.rect_size/2,
                fill='black')

            # create oval
            oval_center = origin + np.array([self.unit * oval_loc_w, self.unit * oval_loc_h])
            self.oval = self.canvas.create_oval(
                oval_center[0] - self.rect_size/2, oval_center[1] - self.rect_size/2,
                oval_center[0] + self.rect_size/2, oval_center[1] + self.rect_size/2,
                fill='yellow')

            # create agents
            for i in range(self.n_agents):
                self.rect_list.append(self.canvas.create_rectangle(
                    origin[0] + self.unit*(self.rect_pos_list[i*2]-1) - self.rect_size/2, origin[1] + self.unit*(self.rect_pos_list[i*2+1]-1) - self.rect_size/2,
                    origin[0] + self.unit*(self.rect_pos_list[i*2]-1) + self.rect_size/2, origin[1] + self.unit*(self.rect_pos_list[i*2+1]-1) + self.rect_size/2,
                    fill='red'))


        # pack all
            self.canvas.pack()

    def reset(self):
        self.last_dis = 0
        self.done = False
        self.rect_pos_list = self.rect_pos_list_his.copy()
        hell1_loc_h = random.randint(0, self.maze_h-1)
        hell1_loc_w = random.randint(0, self.maze_w-1)
        oval_loc_h = random.randint(0, self.maze_h-1)
        oval_loc_w = random.randint(0, self.maze_w-1)
        if (hell1_loc_h == oval_loc_h and hell1_loc_w == oval_loc_w):
            oval_loc_h += 1
        if oval_loc_h > self.maze_h - 1:
            oval_loc_h = oval_loc_h - self.maze_h
        for i in range(self.n_agents):
            if i*self.n_features >= len(self.rect_pos_list):
                loc_h = random.randint(0, self.maze_h-1)
                loc_w = random.randint(0, self.maze_w-1)
                self.rect_pos_list.append(loc_w)
                self.rect_pos_list.append(loc_h)

        if self.show:
            self.update()
        #time.sleep(0.1)
            self.canvas.delete(self.oval)
            self.canvas.delete(self.hell1)
            for i in range(self.n_agents):
                temp = self.rect_list.pop()
                self.canvas.delete(temp)

            origin = np.array([self.unit/2, self.unit/2])
            hell1_center = origin + np.array([self.unit * hell1_loc_w, self.unit * hell1_loc_h])
            self.hell1 = self.canvas.create_rectangle(
                hell1_center[0] - self.rect_size/2, hell1_center[1] - self.rect_size/2,
                hell1_center[0] + self.rect_size/2, hell1_center[1] + self.rect_size/2,
                fill='black')

            oval_center = origin + np.array([self.unit * oval_loc_w, self.unit * oval_loc_h])
            self.oval = self.canvas.create_oval(
                oval_center[0] - self.rect_size/2, oval_center[1] - self.rect_size/2,
                oval_center[0] + self.rect_size/2, oval_center[1] + self.rect_size/2,
                fill='yellow')


            # create agents
            for i in range(self.n_agents):
                self.rect_list.append(self.canvas.create_rectangle(
                    origin[0] + self.unit*(self.rect_pos_list[i*2]-1) - self.rect_size/2, origin[1] + self.unit*(self.rect_pos_list[i*2+1]-1) - self.rect_size/2,
                    origin[0] + self.unit*(self.rect_pos_list[i*2]-1) + self.rect_size/2, origin[1] + self.unit*(self.rect_pos_list[i*2+1]-1) + self.rect_size/2,
                    fill='red'))

        # return observation
        
        return self.rect_pos_list

    """
    def env_s(self):
        return np.hstack(
                        ((np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT),
                         (np.array(self.canvas.coords(self.rect2)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT),
                         (np.array(self.canvas.coords(self.rect3)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
                        )
                        )
    """

    def step(self, action):
        for i in range(self.n_agents):
            base_action = np.array([0, 0])
            if action[i] == 0:   # up
                if self.rect_pos_list[i*2+1] > 0:
                    base_action[1] -= 1
                else:
                    base_action[1] += self.maze_h-1
            elif action[i] == 1:   # down
                if self.rect_pos_list[i*2+1] < self.maze_h:
                    base_action[1] += 1
                else:
                    base_action[1] -= self.maze_h-1
            elif action[i] == 2:   # right
                if self.rect_pos_list[i*2] < self.maze_w:
                    base_action[0] += 1
                else:
                    base_action[0] -= self.maze_w - 1
            elif action[i] == 3:   # left
                if self.rect_pos_list[i*2] > 0:
                    base_action[0] -= 1
                else:
                    base_action[0] += self.maze_w - 1
            self.rect_pos_list[i*2] += base_action[0]
            self.rect_pos_list[i*2+1] += base_action[1]
            if self.show:
                self.canvas.move(self.rect_list[i], base_action[0]*self.unit, base_action[1]*self.unit)  # move agent


        # reward function
        '''if next_coords == self.canvas.coords(self.oval) or next_coords2 == self.canvas.coords(self.oval) \
                or next_coords3 == self.canvas.coords(self.oval):
            reward = 5
            done = True
        #elif next_coords in [self.canvas.coords(self.hell1)] or next_coords2 in [self.canvas.coords(self.hell1)]\
         #       or next_coords3 in [self.canvas.coords(self.hell1)]:
         #   reward = -2
         #   done = False
        #elif next_coords == next_coords2 or next_coords2 == next_coords3 or next_coords3 == next_coords:
         #   reward = -1
         #   done = False
        else:
            reward = 0
            done = False'''
        '''
        reward = 0
        done = False
        #pdb.set_trace()
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i < j and next_coords[i][0]==next_coords[j][0] and abs(next_coords[i][1] - next_coords[j][1]) == self.unit:
                    reward = reward + 1
                if i < j and next_coords[i][0]==next_coords[j][0] and next_coords[i][1] < self.unit and next_coords[j][1] > self.unit*(self.maze_h-1):
                    reward = reward + 1
                if i < j and next_coords[i][0]==next_coords[j][0] and next_coords[j][1] < self.unit and next_coords[i][1] > self.unit*(self.maze_h-1):
                    reward = reward + 1
                if i < j and next_coords[i][1]==next_coords[j][1] and abs(next_coords[i][0] - next_coords[j][0]) == self.unit:
                    reward = reward + 1
                if i < j and next_coords[i][1]==next_coords[j][1] and next_coords[i][0] < self.unit and next_coords[j][0] > self.unit*(self.maze_w-1):
                    reward = reward + 1
                if i < j and next_coords[i][1]==next_coords[j][1] and next_coords[j][0] < self.unit and next_coords[i][0] > self.unit*(self.maze_w-1):
                    reward = reward + 1
        '''

        '''
        reward = self.n_agents*(self.n_agents - 1) - 1
        done = True
        #pdb.set_trace()
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if next_coords[i][0]!=next_coords[j][0] or next_coords[i][1]!=next_coords[j][1]:
                    reward = reward - 1
                    done = False

        if done:
            reward = self.n_agents*self.n_agents*300

        '''

        dis = 0
        done = True
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                dis = dis + abs(self.rect_pos_list[i*2]-self.rect_pos_list[j*2]) + abs(self.rect_pos_list[i*2+1]-self.rect_pos_list[j*2+1])
                if self.rect_pos_list[i*2]!=self.rect_pos_list[j*2] or self.rect_pos_list[i*2+1]!=self.rect_pos_list[j*2+1]:
                    done = False
        reward = self.last_dis - dis - self.n_agents*self.n_agents
        self.last_dis = dis

        #if done:
            #reward = self.n_agents*self.n_agents*self.maze_h*self.maze_w*100

        if self.done:
            done_final = True
        else:
            done_final = False
        self.done = done

        '''#print(next_coords[0],next_coords2[0],next_coords3[0],(MAZE_H - 1) * UNIT,UNIT)
        if next_coords[0] > (MAZE_H - 1) * UNIT or next_coords2[0] > (MAZE_H - 1) * UNIT:
            reward = -0.1
            #print('q')
            done = False
        elif next_coords[0] < UNIT or next_coords2[0] < UNIT:
            reward = -0.1
            #print('w')
            done = False
        elif next_coords[1] > (MAZE_H - 1) * UNIT or next_coords2[1] > (MAZE_H - 1) * UNIT:
            reward = -0.1
            #print('e')
            done = False
        elif next_coords[1] < UNIT or next_coords2[1] < UNIT:
            reward = -0.1
            #print('r')
            done = False
        elif next_coords[0] == next_coords2[0] and next_coords3[0] == next_coords2[0]:
            reward = 0.4
            #print(1)
            done = False
        elif next_coords[0] == next_coords2[0]:
            reward = 0.2
            done = False
        elif next_coords3[0] == next_coords2[0]:
            reward = 0.2
            #print(2)
            done = False
        elif next_coords3[0] == next_coords[0]:
            reward = 0.2
            #print(3)
            done = False
        else:
            reward = -0.1
            done = False'''

        return self.rect_pos_list, reward, done_final

    def render(self):
        #time.sleep(0.01)
        if self.show:
            self.update()


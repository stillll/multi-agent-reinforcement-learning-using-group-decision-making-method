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
                 unit = 100,
                 maze_h = 3,
                 maze_w = 1,
                 rect_size = 80,
                 rect_pos = []
                 ):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.n_agents = n_agents
        self.max_coop = max_coop
        self.unit = unit
        self.maze_h = maze_h
        self.maze_w = maze_w
        self.rect_size = rect_size
        self.rect_pos_list = rect_pos
        self.rect_list = []
        self.title('maze')
        self.geometry('{0}x{1}'.format(self.maze_w * self.unit, self.maze_h * self.unit))
        self._build_maze()

    def _build_maze(self,):
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
        loc_h = random.randint(0, self.maze_h-1)
        loc_w = random.randint(0, self.maze_w-1)
        hell1_center = origin + np.array([self.unit * loc_w, self.unit * loc_h])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - self.rect_size/2, hell1_center[1] - self.rect_size/2,
            hell1_center[0] + self.rect_size/2, hell1_center[1] + self.rect_size/2,
            fill='black')
        # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # create oval
        loc_h = random.randint(0, self.maze_h-1)
        loc_w = random.randint(0, self.maze_w-1)
        oval_center = origin + np.array([self.unit * loc_w, self.unit * loc_h])
        if (oval_center.all == hell1_center).all():
            oval_center += self.unit
        if oval_center[0] > self.unit*self.maze_h:
            oval_center[0] = oval_center[0] - self.unit*self.maze_h
        if oval_center[1] > self.unit*self.maze_w:
            oval_center[1] = oval_center[1] - self.unit*self.maze_w
        self.oval = self.canvas.create_oval(
            oval_center[0] - self.rect_size/2, oval_center[1] - self.rect_size/2,
            oval_center[0] + self.rect_size/2, oval_center[1] + self.rect_size/2,
            fill='yellow')

        for i in range(self.n_agents):
            if i*2 < len(self.rect_pos_list):
                self.rect_list.append(self.canvas.create_rectangle(
                    origin[0] + self.unit*(self.rect_pos_list[i*2]-1) - self.rect_size/2, origin[1] + self.unit*(self.rect_pos_list[i*2+1]-1) - self.rect_size/2,
                    origin[0] + self.unit*(self.rect_pos_list[i*2]-1) + self.rect_size/2, origin[1] + self.unit*(self.rect_pos_list[i*2+1]-1) + self.rect_size/2,
                    fill='red'))
            else:
                loc_h = random.randint(0, self.maze_h-1)
                loc_w = random.randint(0, self.maze_w-1)
                oval_center = origin + np.array([self.unit * loc_w, self.unit * loc_h])
                self.rect_list.append(self.canvas.create_rectangle(
                    oval_center[0] - self.rect_size/2, oval_center[1] - self.rect_size/2,
                    oval_center[0] + self.rect_size/2, oval_center[1] + self.rect_size/2,
                    fill='red'))

        '''
        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - self.rect_size/2, origin[1] - self.rect_size/2,
            origin[0] + self.rect_size/2, origin[1] + self.rect_size/2,
            fill='red')
        pdb.set_trace()
        
        # create green rect
        self.rect2 = self.canvas.create_rectangle(
            origin[0] + self.unit*(self.maze_w-1) - self.rect_size/2, origin[1] - self.rect_size/2,
            origin[0] + self.unit*(self.maze_w-1) + self.rect_size/2, origin[1] + self.rect_size/2,
            fill='green')

        # create blue rect
        self.rect3 = self.canvas.create_rectangle(
            origin[0] + self.unit*(self.maze_w-1) - self.rect_size/2, origin[1] + self.unit*(self.maze_h-1) - self.rect_size/2,
            origin[0] + self.unit*(self.maze_w-1) + self.rect_size/2, origin[1] + self.unit*(self.maze_h-1) + self.rect_size/2,
            fill='blue')
'''
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        #time.sleep(0.1)
        self.canvas.delete(self.oval)
        self.canvas.delete(self.hell1)
        for i in range(self.n_agents):
            temp = self.rect_list.pop()
            self.canvas.delete(temp)
        origin = np.array([self.unit/2, self.unit/2])
        loc_h = random.randint(0, self.maze_h-1)
        loc_w = random.randint(0, self.maze_w-1)
        hell1_center = origin + np.array([self.unit * loc_w, self.unit * loc_h])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - self.rect_size/2, hell1_center[1] - self.rect_size/2,
            hell1_center[0] + self.rect_size/2, hell1_center[1] + self.rect_size/2,
            fill='black')
        loc_h = random.randint(0, self.maze_h-1)
        loc_w = random.randint(0, self.maze_w-1)
        oval_center = origin + np.array([self.unit * loc_w, self.unit * loc_h])
        if (oval_center.all == hell1_center).all():
            oval_center += self.unit
        if oval_center[0] > self.unit*self.maze_h:
            oval_center[0] = oval_center[0] - self.unit*self.maze_h
        if oval_center[1] > self.unit*self.maze_w:
            oval_center[1] = oval_center[1] - self.unit*self.maze_w
        self.oval = self.canvas.create_oval(
            oval_center[0] - self.rect_size/2, oval_center[1] - self.rect_size/2,
            oval_center[0] + self.rect_size/2, oval_center[1] + self.rect_size/2,
            fill='yellow')


        for i in range(self.n_agents):
            if i*2 < len(self.rect_pos_list):
                self.rect_list.append(self.canvas.create_rectangle(
                    origin[0] + self.unit*(self.rect_pos_list[i*2]-1) - self.rect_size/2, origin[1] + self.unit*(self.rect_pos_list[i*2+1]-1) - self.rect_size/2,
                    origin[0] + self.unit*(self.rect_pos_list[i*2]-1) + self.rect_size/2, origin[1] + self.unit*(self.rect_pos_list[i*2+1]-1) + self.rect_size/2,
                    fill='red'))
            else:
                loc_h = random.randint(0, self.maze_h-1)
                loc_w = random.randint(0, self.maze_w-1)
                oval_center = origin + np.array([self.unit * loc_w, self.unit * loc_h])
                self.rect_list.append(self.canvas.create_rectangle(
                    oval_center[0] - self.rect_size/2, oval_center[1] - self.rect_size/2,
                    oval_center[0] + self.rect_size/2, oval_center[1] + self.rect_size/2,
                    fill='red'))
        '''
        self.rect = self.canvas.create_rectangle(
            origin[0] - self.rect_size/2, origin[1] - self.rect_size/2,
            origin[0] + self.rect_size/2, origin[1] + self.rect_size/2,
            fill='red')
        self.rect2 = self.canvas.create_rectangle(
            origin[0] + self.unit*(self.maze_w-1) - self.rect_size/2, origin[1] - self.rect_size/2,
            origin[0] + self.unit*(self.maze_w-1) + self.rect_size/2, origin[1] + self.rect_size/2,
            fill='green')
        self.rect3 = self.canvas.create_rectangle(
            origin[0] + self.unit*(self.maze_w-1) - self.rect_size/2, origin[1] + self.unit*(self.maze_h-1) - self.rect_size/2,
            origin[0] + self.unit*(self.maze_w-1) + self.rect_size/2, origin[1] + self.unit*(self.maze_h-1) + self.rect_size/2,
            fill='blue')
        '''
        # return observation
        env_s = []
        for i in range(self.n_agents):
            env_s.append((self.canvas.coords(self.rect_list[i])[0]+self.canvas.coords(self.rect_list[i])[2])/2 + self.unit/2)
            env_s.append((self.canvas.coords(self.rect_list[i])[1]+self.canvas.coords(self.rect_list[i])[3])/2 + self.unit/2)
        env_s = np.array(env_s)/self.unit
        env_s = np.hstack(env_s)
        return env_s[:self.n_agents*self.n_features]

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
        next_coords = []

        for i in range(self.n_agents):
            base_action = np.array([0, 0])
            if action[i] == 0:   # up
                if self.canvas.coords(self.rect_list[i])[1] > self.unit:
                    base_action[1] -= self.unit
                else:
                    base_action[1] += self.unit*(self.maze_h-1)
            elif action[i] == 1:   # down
                if self.canvas.coords(self.rect_list[i])[1] < (self.maze_h - 1) * self.unit:
                    base_action[1] += self.unit
                else:
                    base_action[1] -= self.unit*(self.maze_h-1)
            elif action[i] == 2:   # right
                if self.canvas.coords(self.rect_list[i])[0] < (self.maze_w - 1) * self.unit:
                    base_action[0] += self.unit
                else:
                    base_action[0] -= self.unit*(self.maze_w-1)
            elif action[i] == 3:   # left
                if self.canvas.coords(self.rect_list[i])[0] > self.unit:
                    base_action[0] -= self.unit
                else:
                    base_action[0] += self.unit*(self.maze_w-1)
            self.canvas.move(self.rect_list[i], base_action[0], base_action[1])  # move agent
            next_coords.append([(self.canvas.coords(self.rect_list[i])[0]+self.canvas.coords(self.rect_list[i])[2])/2 + self.unit/2
                ,(self.canvas.coords(self.rect_list[i])[1]+self.canvas.coords(self.rect_list[i])[3])/2 + self.unit/2])  # next state

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


        reward = 5
        done = True
        #pdb.set_trace()
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if next_coords[i][0]!=next_coords[j][0] or next_coords[i][1]!=next_coords[j][1]:
                    reward = -1
                    done = False


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
        env_s = []
        for i in range(self.n_agents):
            env_s.append(np.array(next_coords[i][:2])/self.unit)
        env_s = np.hstack(env_s)
        return env_s[:self.n_agents*self.n_features], reward, done

    def render(self):
        #time.sleep(0.01)
        self.update()


import numpy as np
import time
import sys
import random




class Maze():
    def __init__(self,
                 n_agents=3,
                 maze_h = 9,
                 maze_w = 9
                 ):
        self.maze_h = maze_h
        self.maze_w = maze_w
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.n_agents = n_agents
        aimposx = random.randint(0, self.maze_w-1)
        aimposy = random.randint(0, self.maze_h-1)
        self.aimpos = np.array([aimposx,aimposy])
        self.agent1 = np.array([0,0])
        self.agent2 = np.array([0,self.maze_h-1])
        self.agent3 = np.array([self.maze_w-1,0])
        self.agent4 = np.array([self.maze_w-1,self.maze_h-1])

        # If the number of agents is biger then 4, add agentlist here.



    def reset(self):
        aimposx = random.randint(0, self.maze_w-1)
        aimposy = random.randint(0, self.maze_h-1)
        self.aimpos = np.array([aimposx,aimposy])
        self.agent1 = np.array([0,0])
        self.agent2 = np.array([0,self.maze_h-1])
        self.agent3 = np.array([self.maze_w-1,0])
        self.agent4 = np.array([self.maze_w-1,self.maze_h-1])
        # return observation
        return np.hstack((self.agent1-self.aimpos,
                         self.agent2-self.aimpos,
                         self.agent3-self.aimpos,
                         self.agent4-self.aimpos)).astype(float)

    def step(self, action):
        for i in range(self.n_agents):
            if i == 0:
                s = self.agent1
            elif i == 1:
                s = self.agent2
            elif i == 2:
                s = self.agent3
            elif i == 3:
                s = self.agent4
            base_action = np.array([0, 0])
            if action[i] == 0:   # up
                if s[1] > 0:
                    base_action[1] -= 1
            elif action[i] == 1:   # down
                if s[1] < self.maze_h - 1:
                    base_action[1] += 1
            elif action[i] == 2:   # right
                if s[0] < self.maze_w - 1:
                    base_action[0] += 1
            elif action[i] == 3:   # left
                if s[0] > 0:
                    base_action[0] -= 1
            if i == 0:
                self.agent1 = self.agent1 + base_action  # move agent
            elif i == 1:
                self.agent2 = self.agent2 + base_action
            elif i == 2:
                self.agent3 = self.agent3 + base_action
            elif i == 3:
                self.agent4 = self.agent4 + base_action

        # reward function
        if all(self.agent1 == self.aimpos) or all(self.agent2 == self.aimpos) \
                or all(self.agent3 == self.aimpos) or all(self.agent4 == self.aimpos):
            reward = 5
            done = True
        elif all(self.agent1 == self.agent2) or all(self.agent1 == self.agent3) or all(self.agent2 == self.agent3) \
            or all(self.agent1 == self.agent4) or all(self.agent2 == self.agent4) or all(self.agent3 == self.agent4):
            reward = -1
            done = False
        else:
            reward = -0.1
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
        env_s_ = np.hstack((self.agent1-self.aimpos,
                         self.agent2-self.aimpos,
                         self.agent3-self.aimpos,
                         self.agent4-self.aimpos)).astype(float)
        return env_s_, reward, done


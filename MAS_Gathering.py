#!/usr/bin/env python3
# encoding=utf-8


import numpy as np
import scipy.misc


class AgentObj:
    def __init__(self, coordinates, type, name, direction=0, mark=0, blood=1):
        self.x = coordinates[0]
        self.y = coordinates[1]
        # 0: r, 1: g, 3: b
        self.type = type
        self.name = name
        self.blood = blood

        # 0: right, 1:top 2: left. 3: bottom
        self.direction = direction
        self.mark = mark

    def is_blood(self):
        return self.blood < 0

    def add_mark(self, agent_blood):
        pass
        # self.mark += 1
        # if self.mark >= 2:
        #     self.mark = 0
        #     self.blood = agent_blood
        # return self.mark

    def sub_blood(self):
        # self.blood -= 1
        self.blood = 0 if self.blood <= 0 else self.blood
        return self.blood

    def turn_left(self, **kwargs):
        self.direction = (self.direction + 1) % 4
        return self.direction

    def turn_right(self, **kwargs):
        self.direction = (self.direction - 1 + 4) % 4
        return self.direction

    def move_forward_delta(self):
        if self.direction == 0:
            delta_x, delta_y = 1, 0
        elif self.direction == 1:
            delta_x, delta_y = 0, -1
        elif self.direction == 2:
            delta_x, delta_y = -1, 0
        elif self.direction == 3:
            delta_x, delta_y = 0, 1
        else:
            assert self.direction in range(4), 'wrong direction'

        return delta_x, delta_y

    def move_left_delta(self):
        if self.direction == 0:
            delta_x, delta_y = 0, -1
        elif self.direction == 1:
            delta_x, delta_y = -1, 0
        elif self.direction == 2:
            delta_x, delta_y = 0, 1
        elif self.direction == 3:
            delta_x, delta_y = 1, 0
        else:
            assert self.direction in range(4), 'wrong direction'

        return delta_x, delta_y

    def move_forward(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_forward_delta()

        self.x = self.x + delta_x if self.x + delta_x >=0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >=0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_backward(self, env_x_size, env_y_size):
        forward_delta_x, forward_delta_y = self.move_forward_delta()
        delta_x, delta_y = -forward_delta_x, -forward_delta_y

        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_left(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_left_delta()

        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_right(self, env_x_size, env_y_size):
        left_delta_x, left_delta_y = self.move_left_delta()
        delta_x, delta_y = -left_delta_x, -left_delta_y

        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def stay(self, **kwargs):
        pass

    def beam(self, env_x_size, env_y_size):
        if self.direction == 0:
            beam_set = [(i + 1, self.y) for i in range(self.x, env_x_size - 1)]
        elif self.direction == 1:
            beam_set = [(self.x, i - 1) for i in range(self.y, 0, -1)]
        elif self.direction == 2:
            beam_set = [(i - 1, self.y) for i in range(self.x, 0, -1)]
        elif self.direction == 3:
            beam_set = [(self.x, i + 1) for i in range(self.y, env_y_size - 1)]
        else:
            assert self.direction in range(4), 'wrong direction'
        return beam_set


class FoodObj:
    def __init__(self, coordinates, type=1, blood=1, reward=1):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.type = type
        self.blood = blood
        self.reward = reward

    # def is_blood(self):
        # return self.blood > 0
    def is_blood(self):
        return self.blood <= 0

    def eat(self, food_blood):
        # self.blood = food_blood
        return self.reward

    def sub_blood(self):
        self.blood -= 1
        self.blood = 0 if self.blood <= 0 else self.blood
        return self.blood


class GameEnv:
    def __init__(self, widht=31, hight=11, agent_blood=5, food_blood=4):
        self.size_x = widht
        self.size_y = hight
        self.objects = []
        self.agent_blood = agent_blood
        self.food_blood = food_blood

        # 0: forward, 1: backward, 2: left, 3: right
        # 4: trun lelf, 5:turn right, 6: beam, 7: stay
        self.action_num = 8

        self.reset()

    # 初始化agent位置和food位置，都是固定的
    def reset(self):
        self.agent1 = AgentObj(coordinates=(0, 5), type=2, name='agent1')
        self.agent2 = AgentObj(coordinates=(30, 5), type=0, name='agent2', direction=2)
        self.agent1_actions = [self.agent1.move_forward, self.agent1.move_backward, self.agent1.move_left, self.agent1.move_right,
                               self.agent1.turn_left, self.agent1.turn_right, self.agent1.beam, self.agent1.stay]
        self.agent2_actions = [self.agent2.move_forward, self.agent2.move_backward, self.agent2.move_left, self.agent2.move_right,
                               self.agent2.turn_left, self.agent2.turn_right, self.agent2.beam, self.agent2.stay]
        self.agent1_beam_set = []
        self.agent2_beam_set = []

        self.food_objects = []

        # for x in range(13, 18):
        #     delta = x - 13 if x -13 < 17 - x else 17 -x
        #     self.food_objects.append(FoodObj(coordinates=(x, 5)))
        #     for i in range(delta):
        #         self.food_objects.append(FoodObj(coordinates=(x, 4 - i)))
        #         self.food_objects.append(FoodObj(coordinates=(x, 6 + i)))
        self.food_objects.append(FoodObj(coordinates=(5, 5)))
    # 输入动作对应代码，输出奖励
    def move(self, agent1_action, agent2_action):
        assert agent1_action in range(8), 'agent1 take wrong action'
        assert agent2_action in range(8), 'agent2 take wrong action'

        agent1_old_x, agent1_old_y = self.agent1.x, self.agent1.y
        agent2_old_x, agent2_old_y = self.agent2.x, self.agent2.y

        self.agent1.sub_blood()
        self.agent2.sub_blood()

        self.agent1_beam_set = []
        self.agent2_beam_set = []
        if not self.agent1.is_blood():
            agent1_action_return = self.agent1_actions[agent1_action](env_x_size=self.size_x, env_y_size=self.size_y)
            self.agent1_beam_set = [] if agent1_action != 6 else agent1_action_return
        if not self.agent2.is_blood():
            agent2_action_return = self.agent2_actions[agent2_action](env_x_size=self.size_x, env_y_size=self.size_y)
            self.agent2_beam_set = [] if agent2_action != 6 else agent2_action_return

        if not self.agent1.is_blood() and not self.agent2.is_blood() and\
                ((self.agent1.x == self.agent2.x and self.agent1.y == self.agent2.y) or
                     (self.agent1.x == agent2_old_x and self.agent1.y == agent2_old_y and
                              self.agent2.x == agent1_old_x and self.agent2.y == agent1_old_y)):

            self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
            self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y

        agent1_reward = 0
        agent2_reward = 0
        food_blood = []
        for food in self.food_objects:
            # food.sub_blood()
            if not food.is_blood():
                if not self.agent1.is_blood() and food.x == self.agent1.x and food.y == self.agent1.y:
                    agent1_reward = food.eat(self.food_blood)
                    food.sub_blood()
                    food_blood.append(food.sub_blood())
                elif not self.agent2.is_blood() and food.x == self.agent2.x and food.y == self.agent2.y:
                    agent2_reward = food.eat(self.food_blood)
                    food.sub_blood()
                    food_blood.append(food.sub_blood())

        food_blood = np.array(food_blood)
        if np.all(food_blood == 0):
            done = True
        else:
            done = False

        if (self.agent1.x, self.agent1.y) in self.agent2_beam_set:
            self.agent1.add_mark(self.agent_blood)
        if (self.agent2.x, self.agent2.y) in self.agent1_beam_set:
            self.agent2.add_mark(self.agent_blood)

        return agent1_reward, agent2_reward, done

    # 设置背景颜色；激光轨迹颜色；food颜色；agent及agent移动轨迹的颜色
    def contribute_metrix(self):
        a = np.ones([self.size_y + 2, self.size_x + 2, 3])  # a是RGB三层，并且给地图加宽一圈
        a[1:-1, 1:-1, :] = 0  # 将加的一圈之内的原本的地图值设为0，也就是设背景为黑色

        for x, y in self.agent1_beam_set:  # 将激光照射的地方显示为灰色，及0.5，0.5，0.5
            a[y + 1, x + 1, 0] = 0.5
            a[y + 1, x + 1, 1] = 0.5
            a[y + 1, x + 1, 2] = 0.5
        for x, y in self.agent2_beam_set:
            a[y + 1, x + 1, 0] = 0.5
            a[y + 1, x + 1, 1] = 0.5
            a[y + 1, x + 1, 2] = 0.5

        for food in self.food_objects:
            if not food.is_blood():  # 如果food满血，那就显示出颜色，按照他的类型显示
                for i in range(3):
                    a[food.y + 1, food.x + 1, i] = 1 if i == food.type else 0

        for i in range(3):  # 如果满血，就显示颜色，并且显示移动轨迹，为灰色
            if not self.agent1.is_blood():
                delta_x, delta_y = self.agent1.move_forward_delta()
                a[self.agent1.y + 1 + delta_y, self.agent1.x + 1 + delta_x, i] = 0.5
            if not self.agent2.is_blood():
                delta_x, delta_y = self.agent2.move_forward_delta()
                a[self.agent2.y + 1 + delta_y, self.agent2.x + 1 + delta_x, i] = 0.5
            if not self.agent1.is_blood():
                a[self.agent1.y + 1, self.agent1.x + 1, i] = 1 if i == self.agent1.type else 0
            if not self.agent2.is_blood():
                a[self.agent2.y + 1, self.agent2.x + 1, i] = 1 if i == self.agent2.type else 0

        return a

    # 将图片放大，便于观看
    def render_env(self):
        a = self.contribute_metrix()
        # 函数作用是调整大小scipy.misc.imresize(img,new_size，interp)即将img变为new_size大小，interp是放大采样方式
        b = scipy.misc.imresize(a[:, :, 0], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')

        a = np.stack([b, c, d], axis=2)
        return a

    # 将图片还原，继续处理
    def train_render(self):
        a = self.contribute_metrix()
        # 函数作用是调整大小scipy.misc.imresize(img,new_size，interp)即将img变为new_size大小，interp是放大采样方式
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')

        a = np.stack([b, c, d], axis=2)
        return a

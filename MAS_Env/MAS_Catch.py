#!/usr/bin/env python3
# encoding=utf-8


import numpy as np
import scipy.misc


class AgentObj:
    def __init__(self, coordinates, type, name, direction, mark, blood, speed):
        self.x = coordinates[0]
        self.y = coordinates[1]
        # 0: r, 1: g, 3: b
        self.type = type
        self.name = name
        self.blood = blood
        self.speed = speed

        # 0: right, 1:top 2: left. 3: bottom
        self.direction = direction
        self.mark = mark

    def is_alive(self):
        return self.blood > 0

    def add_mark(self, agent_blood):
        pass
        # self.mark += 1
        # if self.mark >= 2:
        #     self.mark = 0
        #     self.blood = agent_blood
        # return self.mark

    def is_attacked(self):
        # self.blood -= 1
        self.blood = 0 if self.blood <= 0 else self.blood
        return self.blood

    # maybe dropped
    def turn_left(self, **kwargs):
        self.direction = (self.direction + 1) % 4
        return self.direction

    # maybe dropped
    def turn_right(self, **kwargs):
        self.direction = (self.direction - 1 + 4) % 4
        return self.direction

    def move_forward_delta(self):
        if self.direction == 0:
            delta_x, delta_y = self.speed, 0
        elif self.direction == 1:
            delta_x, delta_y = 0, -self.speed
        elif self.direction == 2:
            delta_x, delta_y = -self.speed, 0
        elif self.direction == 3:
            delta_x, delta_y = 0, self.speed
        else:
            assert self.direction in range(4), 'wrong direction'

        return delta_x, delta_y

    def move_left_delta(self):
        if self.direction == 0:
            delta_x, delta_y = 0, -self.speed
        elif self.direction == 1:
            delta_x, delta_y = -self.speed, 0
        elif self.direction == 2:
            delta_x, delta_y = 0, self.speed
        elif self.direction == 3:
            delta_x, delta_y = self.speed, 0
        else:
            assert self.direction in range(4), 'wrong direction'

        return delta_x, delta_y

    def move_forward(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_forward_delta()

        self.x = self.x + delta_x if 0 <= self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if 0 <= self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_backward(self, env_x_size, env_y_size):
        forward_delta_x, forward_delta_y = self.move_forward_delta()
        delta_x, delta_y = -forward_delta_x, -forward_delta_y

        self.x = self.x + delta_x if 0 <= self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if 0 <= self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_left(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_left_delta()

        self.x = self.x + delta_x if 0 <= self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if 0 <= self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_right(self, env_x_size, env_y_size):
        left_delta_x, left_delta_y = self.move_left_delta()
        delta_x, delta_y = -left_delta_x, -left_delta_y

        self.x = self.x + delta_x if 0 <= self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if 0 <= self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def stay(self, **kwargs):
        pass

    # def beam(self, env_x_size, env_y_size):
    #     if self.direction == 0:
    #         beam_set = [(i + 1, self.y) for i in range(self.x, env_x_size - 1)]
    #     elif self.direction == 1:
    #         beam_set = [(self.x, i - 1) for i in range(self.y, 0, -1)]
    #     elif self.direction == 2:
    #         beam_set = [(i - 1, self.y) for i in range(self.x, 0, -1)]
    #     elif self.direction == 3:
    #         beam_set = [(self.x, i + 1) for i in range(self.y, env_y_size - 1)]
    #     else:
    #         assert self.direction in range(4), 'wrong direction'
    #     return beam_set


class FoodObj:
    def __init__(self, coordinates, type, blood, reward, speed):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.type = type
        self.blood = blood
        self.reward = reward
        self.speed = speed

    def is_alive(self):
        return self.blood > 0

    def eat(self):
        self.blood -= 1
        self.blood = 0 if self.blood <= 0 else self.blood
        return self.reward

    def the_blood(self):
        return self.blood

    def moving(self, env_x_size, env_y_size):
        act = [-self.speed, self.speed]
        x_or_y = [0, 1]
        _ = np.random.choice(x_or_y)
        delta = np.random.choice(act)
        if _:
            self.x = self.x + delta if 0 <= self.x + delta < env_x_size else self.x
            self.y = self.y
        else:
            self.x = self.x
            self.y = self.y + delta if 0 <= self.y + delta < env_y_size else self.y
        return self.x, self.y


class GameEnv:
    def __init__(self, width, high, agent_num, food_num, agent_blood, food_blood):
        self.size_x = width
        self.size_y = high
        self.agent_objects = []
        self.food_objects = []
        self.agent_actions = []
        self.food_actions = []
        # self.objects = []
        self.agent_blood = agent_blood
        self.food_blood = food_blood
        self.agent_num = agent_num
        self.food_num = food_num

        # 0: forward, 1: backward, 2: left, 3: right 4: turn left, 5:turn right, 6: stay
        self.action_num = 7
        self.reset()

    # 初始化agent位置和food位置，都是固定的
    def reset(self):
        self.agent_objects = []
        self.agent_actions = []
        for i in range(self.agent_num):
            self.agent_objects.append(AgentObj(coordinates
                                               =(np.random.randint(self.size_x-1), np.random.randint(self.size_y-1)),
                                               type=2,
                                               name=i,
                                               blood=1,
                                               speed=1,
                                               mark=0,
                                               direction=np.random.randint(4)
                                               )
                                      )
            self.agent_actions.append([self.agent_objects[i].move_forward,
                                       self.agent_objects[i].move_backward,
                                       self.agent_objects[i].move_left,
                                       self.agent_objects[i].move_right,
                                       self.agent_objects[i].turn_left,
                                       self.agent_objects[i].turn_right,
                                       self.agent_objects[i].stay]
                                      )
        self.food_objects = []
        for j in range(self.food_num):
            self.food_objects.append(FoodObj(coordinates
                                              =(np.random.randint(self.size_x-1), np.random.randint(self.size_y-1)),
                                             type=1,
                                             blood=1,
                                             reward=10,
                                             speed=1
                                             )
                                      )
    # 输入动作对应代码，输出奖励

    def move(self, agent_action_list):
        # assert agent1_action in range(8), 'agent1 take wrong action'
        # assert agent2_action in range(8), 'agent2 take wrong action'
        # self.agent1.is_attacked()
        # self.agent2.is_attacked()

        # I don't add the collision detection

        for agent in self.agent_objects:
            if agent.is_alive():
                self.agent_actions[agent.name][agent_action_list[agent.name]](env_x_size=self.size_x, env_y_size=self.size_y)

        for food in self.food_objects:
            if food.is_alive():
                food.moving(env_x_size=self.size_x, env_y_size=self.size_y)

        reward_list = []
        food_blood = []
        done = True
        for food in self.food_objects:
            if food.is_alive():
                done = False
                for agent in self.agent_objects:
                    if agent.is_alive() and food.x == agent.x and food.y == agent.y:
                        reward_list.append(food.eat())
                        food_blood.append(food.blood)

        # food_blood = np.array(food_blood)
        # if food_blood and np.all(food_blood == 0):
        #     done = True
        # else:
        #     done = False

        return np.array(reward_list), done

    # 设置背景颜色；激光轨迹颜色；food颜色；agent及agent移动轨迹的颜色
    def contribute_matrix(self):
        a = np.ones([self.size_y + 2, self.size_x + 2, 3])  # a是RGB三层，并且给地图加宽一圈
        a[1:-1, 1:-1, :] = 0  # 将加的一圈之内的原本的地图值设为0，也就是设背景为黑色

        for food in self.food_objects:
            if food.is_alive():  # 如果food满血，那就显示出颜色，按照他的类型显示
                for i in range(3):
                    a[food.y + 1, food.x + 1, i] = 1 if i == food.type else 0

        for i in range(3):  # 如果满血，就显示颜色，并且显示移动轨迹，为灰色
            for agent in self.agent_objects:
                if agent.is_alive():
                    a[agent.y + 1, agent.x + 1, i] = 1 if i == agent.type else 0
                    # delta_x, delta_y = agent.move_forward_delta()
                    # a[agent.y + 1 + delta_y, agent.x + 1 + delta_x, i] = 0.5

        return a

    # 将图片放大，便于观看
    def render_env(self):
        a = self.contribute_matrix()
        # 函数作用是调整大小scipy.misc.imresize(img,new_size，interp)即将img变为new_size大小，interp是放大采样方式
        b = scipy.misc.imresize(a[:, :, 0], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')

        a = np.stack([b, c, d], axis=2)
        return a

    # 将图片还原，继续处理
    def train_render(self):
        a = self.contribute_matrix()
        # 函数作用是调整大小scipy.misc.imresize(img,new_size，interp)即将img变为new_size大小，interp是放大采样方式
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')

        a = np.stack([b, c, d], axis=2)
        return a

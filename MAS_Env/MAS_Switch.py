#!/usr/bin/env python3
# encoding=utf-8

import numpy as np
import scipy.misc


class AgentObj:
    def __init__(self, coordinates, type, name, direction, mark, blood, speed, pickup):
        self.x = coordinates[0]
        self.y = coordinates[1]
        # 0: r agent2, 1: g, 2: b agent1
        self.type = type
        self.name = name
        self.blood = blood
        self.speed = speed

        # 0: right, 1:top 2: left. 3: bottom
        self.direction = direction
        self.mark = mark
        # 0: without, 1: take
        self.pickup = pickup

    def is_pickup(self):
        return self.pickup

    def pick_up(self):
        self.pickup = 1

    def drop_down(self):
        self.pickup = 0

    def is_alive(self):
        return self.blood > 0

    def add_mark(self, agent_blood):
        self.mark += 1
        if self.mark >= 2:
            self.mark = 0
            self.blood = agent_blood
        return self.mark

    def is_attacked(self):
        self.blood -= 1
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

    @staticmethod
    def legal_coordinates(coordinates, env):
        return env.size_x - 1 >= coordinates[0] >= 0 and env.size_y - 1 >= coordinates[1] >= 0 \
               and coordinates not in env.block

    def move_forward(self, env):
        delta_x, delta_y = self.move_forward_delta()

        self.x = self.x + delta_x if self.legal_coordinates([self.x + delta_x, self.y + delta_y], env) else self.x
        self.y = self.y + delta_y if self.legal_coordinates([self.x + delta_x, self.y + delta_y], env) else self.y
        return self.x, self.y

    def move_backward(self, env):
        forward_delta_x, forward_delta_y = self.move_forward_delta()
        delta_x, delta_y = -forward_delta_x, -forward_delta_y

        self.x = self.x + delta_x if self.legal_coordinates([self.x + delta_x, self.y + delta_y], env) else self.x
        self.y = self.y + delta_y if self.legal_coordinates([self.x + delta_x, self.y + delta_y], env) else self.y
        return self.x, self.y

    def move_left(self, env):
        delta_x, delta_y = self.move_left_delta()

        self.x = self.x + delta_x if self.legal_coordinates([self.x + delta_x, self.y + delta_y], env) else self.x
        self.y = self.y + delta_y if self.legal_coordinates([self.x + delta_x, self.y + delta_y], env) else self.y
        return self.x, self.y

    def move_right(self, env):
        left_delta_x, left_delta_y = self.move_left_delta()
        delta_x, delta_y = -left_delta_x, -left_delta_y

        self.x = self.x + delta_x if self.legal_coordinates([self.x + delta_x, self.y + delta_y], env) else self.x
        self.y = self.y + delta_y if self.legal_coordinates([self.x + delta_x, self.y + delta_y], env) else self.y
        return self.x, self.y

    def stay(self, **kwargs):
        pass

    def beam(self, env):  # env_x_size, env_y_size):
        env_x_size = env.size_x
        env_y_size = env.size_y
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
    def __init__(self, coordinates, type, name, reward, blood, speed):
        # # type: 1 is agent1's food, 3 is agent2's food
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.type = type
        self.reward = reward
        self.blood = blood
        self.speed = speed
        self.name = name

    def eat(self):
        self.blood -= 1
        self.blood = 0 if self.blood <= 0 else self.blood
        return self.reward

    def add_blood(self):
        self.blood = 1

    def is_alive(self):
        return self.blood > 0


class GameEnv:
    def __init__(self, width, high, agent_num, food_num, agent_blood, food_blood):
        self.size_x = width
        self.size_y = high
        # self.objects = []
        self.agent_objects = []
        self.food_objects = []
        self.agent_actions = []
        self.food_actions = []
        self.agent_blood = agent_blood
        self.food_blood = food_blood
        self.agent_num = agent_num
        self.food_num = food_num

        # 0: forward, 1: backward, 2: left, 3: right
        # 4: turn left, 5:turn right, 6: beam, 7: stay
        self.action_num = 8
        self.block_f_list = [self.block_level_0, self.block_level_1, self.block_level_2]
        self.agents_beam_set = []

        self.reset()

    def reset(self, block_level=0):
        self.agents_beam_set = []

        block = self.block_f_list[block_level]()
        self.block = tuple(block)
        
        self.agent_objects = []
        self.agent_actions = []
        for i in range(int(self.agent_num/2)):
            self.agent_objects.append(AgentObj(coordinates
                                               =(np.random.randint(int(self.size_x)), np.random.randint(self.size_y)),
                                               type=2,
                                               name=i,
                                               blood=1,
                                               speed=1,
                                               mark=0,
                                               direction=1,  # np.random.randint(4),
                                               pickup=0
                                               )
                                      )
        
            self.agent_actions.append([self.agent_objects[i].move_forward,
                                       self.agent_objects[i].move_backward,
                                       self.agent_objects[i].move_left,
                                       self.agent_objects[i].move_right,
                                       # self.agent_objects[i].turn_left,
                                       # self.agent_objects[i].turn_right,
                                       # self.agent_objects[i].beam,
                                       self.agent_objects[i].stay]
                                      )
        for i in range(int(self.agent_num/2), self.agent_num):
            self.agent_objects.append(AgentObj(coordinates
                                               =(np.random.randint(self.size_x), np.random.randint(self.size_y)),
                                               type=0,
                                               name=i,
                                               blood=1,
                                               speed=1,
                                               mark=0,
                                               direction=1,  # np.random.randint(4),
                                               pickup=0
                                               )
                                      )
            self.agent_actions.append([self.agent_objects[i].move_forward,
                                       self.agent_objects[i].move_backward,
                                       self.agent_objects[i].move_left,
                                       self.agent_objects[i].move_right,
                                       # self.agent_objects[i].turn_left,
                                       # self.agent_objects[i].turn_right,
                                       # self.agent_objects[i].beam,
                                       self.agent_objects[i].stay]
                                      )
        self.food_objects = []
        for j in range(int(self.food_num / 2)):
            self.food_objects.append(FoodObj(coordinates
                                             =(np.random.randint(int(self.size_x)), np.random.randint(self.size_y)),
                                             name=j,
                                             type=1,
                                             blood=1,
                                             reward=1,
                                             speed=0
                                             )
                                     )
        for j in range(int(self.food_num / 2), self.food_num):
            self.food_objects.append(FoodObj(coordinates  # int(4*self.size_x/5),
                                             =(np.random.randint(self.size_x), np.random.randint(self.size_y)),
                                             name=j,
                                             type=3,
                                             blood=0,
                                             reward=1,
                                             speed=0
                                             )
                                     )

    def block_level_0(self):  # 总的背景块
        block = []
        return block

    def block_level_1(self):
        block = []
        for x in range(int(0.2 * self.size_x), int(0.8 * self.size_x)):
            for y in range(int(0.2 * self.size_y), int(0.8 * self.size_y)):
                block.append([x, y])
        return block

    def block_level_2(self):  # 上下两个墙体
        block = []
        for x in range(int(0.2 * self.size_x), int(0.8 * self.size_x)):
            for y in range(0, 3):
                block.append([x, y])
                block.append([x, y + 16])
                block.append([x, y + 31])
        return block

    # def check_env_done(self):
    #     return self.food_objects[0].is_alive() and self.food_objects[1].is_alive()

    def move(self, agent_action_list):
        # assert agent1_action in range(8), 'agent1 take wrong action'
        # assert agent2_action in range(8), 'agent2 take wrong action'

        agents_old_x_y = []
        for agent in self.agent_objects:
            agents_old_x_y.append([agent.x, agent.y])
            # agent.is_attacked()
            # temp_act_return = \
            self.agent_actions[agent.name][agent_action_list[agent.name]](env=self)
            # self.agents_beam_set.append([] if agent_action_list[agent.name] != 6 else temp_act_return)

        # agent1_action_return = self.agent1_actions[agent1_action](env=self)
        # self.agent1_beam_set = [] if agent1_action != 6 else agent1_action_return
        #
        # agent2_action_return = self.agent2_actions[agent2_action](env=self)
        # #env_x_size=self.size_x, env_y_size=self.size_y)
        # self.agent2_beam_set = [] if agent2_action != 6 else agent2_action_return

        for i in self.agent_objects:
            for j in self.agent_objects:
                if i == j:
                    pass
                else:
                    if i.x == j.x and i.y == j.y:
                        i.x, i.y = agents_old_x_y[i.name]
                        j.x, j.y = agents_old_x_y[j.name]

        reward_list = []
        food_blood = []

        # 应该以agent进行遍历，因为每个step每个agent都只需要得到一个奖惩值；
        # 而如果对food进行遍历，每个agent重复得到reward是不正确的。
        done = True
        for agent in self.agent_objects:
            reward = -0.01
            for food in self.food_objects:
                if food.is_alive():
                    done = False
                    if food.x == agent.x and food.y == agent.y:
                        if agent.type == 2 and food.type == 1:
                            reward = food.eat()
                            self.food_objects[food.name + int(self.food_num / 2)].add_blood()
                        elif agent.type == 0 and food.type == 3:
                            reward = food.eat()
                            self.food_objects[food.name - int(self.food_num / 2)].add_blood()
                        # elif agent.type == 2 and food.type == 3:
                        #     reward = -0.1
                        # elif agent.type == 0 and food.type == 1:
                        #     reward = -0.1
            reward_list.append(reward)
            food_blood.append(food.blood)

        return np.array(reward_list), done

    def contribute_matrix(self):
        a = np.zeros([self.size_y + 2, self.size_x + 2, 3], dtype=np.float32)
        # a[1:-1, 1:-1, :] = 0
        # 画出边框颜色
        a[:, 0, 0] = 136 / 255
        a[:, 0, 1] = 136 / 255
        a[0, 0, 1] = 255 / 255
        a[:, 0, 2] = 135 / 255

        a[:, self.size_x + 1, 0] = 136 / 255
        a[:, self.size_x + 1, 1] = 136 / 255
        a[:, self.size_x + 1, 2] = 135 / 255

        a[0, :, 0] = 136 / 255
        a[0, :, 1] = 136 / 255
        a[0, :, 2] = 135 / 255

        a[self.size_y + 1, :, 0] = 136 / 255
        a[self.size_y + 1, :, 1] = 136 / 255
        a[self.size_y + 1, :, 2] = 136 / 255

        for block in self.block:
            a[block[1] + 1, block[0] + 1, 0] = 0.53  # 136 / 255
            a[block[1] + 1, block[0] + 1, 1] = 0.54  # 138 / 255
            a[block[1] + 1, block[0] + 1, 2] = 0.53  # 136 / 255

        # 画出激光颜色
        # print(self.agents_beam_set)
        # for x, y in self.agents_beam_set:
        #     a[y + 1, x + 1, 0] = 0.5
        #     a[y + 1, x + 1, 1] = 0.5
        #     a[y + 1, x + 1, 2] = 0.5

        # 画出还存活的不同food的图像
        for food in self.food_objects:
            if food.is_alive():
                if food.type == 1:
                    a[food.y + 1, food.x + 1, 0] = 12 / 255
                    a[food.y + 1, food.x + 1, 1] = 255 / 255
                    a[food.y + 1, food.x + 1, 2] = 134 / 255
                elif food.type == 3:
                    a[food.y + 1, food.x + 1, 0] = 117 / 255
                    a[food.y + 1, food.x + 1, 1] = 255 / 255
                    a[food.y + 1, food.x + 1, 2] = 0

        for i in range(3):
            for agent in self.agent_objects:
                a[agent.y + 1, agent.x + 1, i] = 1 if i == agent.type else 0
        return a

    def render_env(self):
        a = self.contribute_matrix()

        b = scipy.misc.imresize(a[:, :, 0], [10 * self.size_y, 10 * self.size_x, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [10 * self.size_y, 10 * self.size_x, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [10 * self.size_y, 10 * self.size_x, 1], interp='nearest')

        a = np.stack([b, c, d], axis=2)
        return a

    def train_render(self):
        a = self.contribute_matrix()

        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')

        a = np.stack([b, c, d], axis=2)
        return a

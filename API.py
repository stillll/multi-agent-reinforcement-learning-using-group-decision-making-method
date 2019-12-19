import argparse
from WAgent.python import magent


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="pursuit", help="name of the scenario script")
    parser.add_argument("--map_size", type=int, default=100, help="the size of map")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c', 'gdm'])
    parser.add_argument('--max_step', type=int, default=300, help='the max step of a episode')
    return parser.parse_args()


def make_env(arglist):
    env = magent.GridWorld(arglist.scenario, map_size=arglist.map_size)
    env.set_render_dir("build/render")
    # get group handles
    predator, prey = env.get_handles()
    # init env and agents
    env.reset()
    env.add_walls(method="random", n=arglist.map_size * arglist.map_size * 0.01)
    env.add_agents(predator, method="random", n=arglist.map_size * arglist.map_size * 0.02)
    env.add_agents(prey, method="random", n=arglist.map_size * arglist.map_size * 0.02)
    return env


def set_model(arglist, env):
    from . import Model
    model = Model.method(env, method=arglist.alg)

    return model


def train(arglist):
    env = make_env(arglist)
    model = set_model(arglist, env)
    predator, prey = env.get_handles()
    done = False
    step = 0
    while not done:
        print("nums: %d vs %d" % (env.get_num(predator), env.get_num(prey)))
        num_1 = env.get_num(predator)  # int type
        alive_1 = env.get_alive(predator)  # bool type, shape: [True, ... , True]
        ids_1 = env.get_agent_id(predator)  # shape: [1, ... , n]
        pos_1 = env.get_pos(predator)  # shape: [[x1, y1], ... , [xn, yn]]
        obs_1 = model.cg(num_1, alive_1, ids_1, pos_1)
        acts_1 = model.take_action(obs_1, ids_1)
        env.set_action(predator, acts_1)
        reward_1 = env.get_reward(predator)  # shape: [1 * n]

        num_2 = env.get_num(prey)
        alive_2 = env.get_alive(prey)
        ids_2 = env.get_agent_id(prey)
        pos_2 = env.get_pos(prey)
        obs_2 = model.cg(num_2, alive_2, ids_2, pos_2)
        acts_2 = model.take_action(obs_2, ids_2)
        env.set_action(prey, acts_2)
        reward_2 = env.get_reward(prey)

        # simulate one step
        done = env.step()

        # render
        env.render()

        # get reward
        reward = [sum(reward_1), sum(reward_2)]

        # clear dead agents
        env.clear_dead()

        # print info
        if step % 10 == 0:
            print("step: %d\t predators' reward: %d\t preys' reward: %d" %
                  (step, reward[0], reward[1]))

        step += 1
        if step > arglist.max_step:
            break


if __name__ == "__main__":
    arglist = parse_args()
    train(arglist)

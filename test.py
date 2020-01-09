

def test_model(env, model, max_episode):

    cumulate_reward = 0
    for episode in range(max_episode):
        print("test_episode", episode)
        env_s = env.reset()  # init env and return env state
        # print(env_s)
        counter = 0  # if end episode
        while True:  # one step
            # fresh env
            env.render()

            # all agents join actions for this step
            join_act = model.run_model(env_s)[0]
            # take action and get next env state and reward
            env_s_, reward, done = env.step(join_act)

            cumulate_reward = reward + cumulate_reward * 0.99

            counter += 1
            if counter > 300:
                break

            env_s = env_s_

            # break while loop when end of this episode
            if done:
                break

        print("reward:", cumulate_reward)
        model.reward_his.append(cumulate_reward)

    # end of game
    print('game over')
    env.destroy()
    model.plot_reward()


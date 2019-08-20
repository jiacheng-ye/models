from maze_env import Maze
from brain_AC import AC

# Hyper Parameters
LR = 0.01  # learning rate
GAMMA = 0.9  # reward discount


def update():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)

            RL.learn(observation, action, reward, observation_)

            observation = observation_

            if done:
                break

            step += 1
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    N_ACTIONS = env.n_actions
    N_STATES = env.n_features

    RL = AC(N_ACTIONS, N_STATES, learning_rate=LR, gamma=GAMMA)
    env.after(100, update)
    env.mainloop()
    RL.plot_cost()

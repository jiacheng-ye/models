from maze_env import Maze
from brain_PG import PolicyGradient

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

            RL.store_transition(observation, action, reward)

            observation = observation_

            if done:
                RL.learn()
                break

            step += 1
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    N_ACTIONS = env.n_actions
    N_STATES = env.n_features

    RL = PolicyGradient(N_ACTIONS, N_STATES, learning_rate=LR, gamma=GAMMA)
    env.after(100, update)
    env.mainloop()
    RL.plot_cost()

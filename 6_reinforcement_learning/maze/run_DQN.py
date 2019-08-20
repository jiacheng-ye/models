from maze_env import Maze
from brain_DQN import DQN

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.6  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 200


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

            RL.store_transition(observation, action, reward, observation_)

            if (step > MEMORY_CAPACITY) and (step % 5 == 0):
                RL.learn()

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

    RL = DQN(N_ACTIONS, N_STATES,
             learning_rate=LR,
             reward_decay=GAMMA,
             e_greedy=EPSILON,
             replace_target_iter=TARGET_REPLACE_ITER,
             memory_size=MEMORY_CAPACITY
             )
    env.after(100, update)
    env.mainloop()
    RL.plot_cost()

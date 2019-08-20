# -*-coding:utf8 -*-

from maze_env import Maze
from brain_table import SarsaTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        action = RL.choose_action(str(observation))
        num_steps = 0
        while True:
            num_steps += 1
            # fresh env
            env.render()

            # 下一个action
            action_ = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_
            action = action_

            if done:
                print('Episode %d, num_steps=%d' % (episode + 1, num_steps))
                break

    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()

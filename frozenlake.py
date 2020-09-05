import gym
import random
import numpy as np
import time

env_name = "FrozenLake-v0"
env = gym.make(env_name)

state_space_size = env.observation_space.n
action_space_size = env.action_space.n

reward_count = 0

num_episodes = 100000
max_steps_per_episode = 1000
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.001

q_table = np.zeros((state_space_size , action_space_size))

rewards_all_episodes = []

# Q-Learning algo
for episode in range(num_episodes):
	state = env.reset()

	done = False
	rewards_current_episode = 0

	for step in range(max_steps_per_episode):

		#Exploration-exploitation tradeoff
		exploration_rate_threshold = random.uniform(0,1)
		if exploration_rate_threshold > exploration_rate:
			action = np.argmax(q_table[state,:])
		else:
			action = env.action_space.sample()

		new_state, reward, done, info = env.step(action)

		#Update Q-Table for Q(s,a)
		q_table[state,action] = q_table[state,action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

		state = new_state
		rewards_current_episode += reward

		if done == True:
			break

	#Exploration rate decay
	exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

	rewards_all_episodes.append(rewards_current_episode)

env.render()
print()
print(sum(rewards_all_episodes)/num_episodes)
print()
print(q_table)

'''
0.71539

[[0.57473948 0.46966861 0.45608717 0.47724861]
 [0.36637184 0.34319681 0.3330869  0.51249019]
 [0.40906572 0.39534733 0.41345056 0.49051271]
 [0.24979733 0.3941615  0.39157841 0.47814952]
 [0.59914276 0.45905686 0.30954636 0.43968442]
 [0.         0.         0.         0.        ]
 [0.15714605 0.15381219 0.4137013  0.11232006]
 [0.         0.         0.         0.        ]
 [0.47072416 0.3972914  0.31554892 0.64543263]
 [0.33632673 0.70164591 0.40585278 0.30858332]
 [0.59310832 0.36492342 0.31893519 0.31682986]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.50442568 0.5169699  0.75139261 0.51321796]
 [0.72512469 0.84846097 0.73852871 0.72479397]
 [0.         0.         0.         0.        ]]
'''
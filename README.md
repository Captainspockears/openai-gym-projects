# openaigym-Projects

So I decided to check out OpenAI's gym to train agents to play games! I want to learn what it takes to teach an AI and hopefully get a foundation inorder to replicate the same in real life.



## [FrozenLake-v0](https://github.com/openai/gym/wiki/FrozenLake-v0)

<div align=”center”><img src="https://user-images.githubusercontent.com/46392391/92300748-fa250480-ef7a-11ea-8472-d11cd69d48fb.png" height="250"><div>
<br>

The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. At this time, there's an international frisbee shortage, so it's absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won't always move in the direction you intend.

The surface is described using a grid like the following:
```
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
```

The episode ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and zero otherwise.

### [My solution](frozenlake.py)

I used Q-learning to train the agent through reinforcement learning.

I used the following parameters:
```
num_episodes = 100000
max_steps_per_episode = 1000
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.001
```
My output was:
```
Probability of winning the game: 0.71539

Q-table:

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
 ```

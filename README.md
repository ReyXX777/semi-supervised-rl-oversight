# semi-supervised-rl-oversight
Semi-supervised reinforcement learning for agents learning from sparse true reward signals alongside proxy feedback.
Semi-Supervised Reinforcement Learning for Scalable Oversight

The repository contains a Python implementation of a semi-supervised reinforcement learning setup where an agent learns primarily from sparse true reward signals while interacting with proxy feedback. The approach demonstrates scalable oversight, allowing agents to optimize performance with limited direct supervision.

Overview

The environment simulates a cleaning task in which an agent must remove messes from a grid. True reward is only occasionally available to the agent, while a proxy reward serves as a cheap approximation. The agent uses tabular Q-learning to optimize for the true reward, learning efficiently even with limited supervision.

The framework illustrates techniques relevant to reinforcement learning in real-world scenarios where direct evaluation is expensive or time-consuming. It also highlights potential safety advantages, as agents trained under sparse oversight must rely on proxy signals to avoid reward hacking or unintended behaviors.

Features

Semi-supervised reward learning allows an agent to combine occasional accurate supervision with frequent approximations.
The Q-learning implementation uses epsilon-greedy exploration with decay to balance exploration and exploitation.
Visualization of cumulative true reward and observed proxy reward per episode provides insight into learning progress.

Requirements

Python 3.8 or higher
NumPy
Matplotlib
TQDM

Install dependencies with:

pip install numpy matplotlib tqdm

Usage

Run the training and visualize results by executing:

python train_semi_supervised_rl.py


The script trains the agent for a number of episodes, plots smoothed cumulative true reward and proxy reward, and demonstrates the learning effect of sparse supervision.

Applications

The code can serve as a toy example for scalable oversight in autonomous systems, efficient learning from limited human feedback, or research in semi-supervised reinforcement learning.

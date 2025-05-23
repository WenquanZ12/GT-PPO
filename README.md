# A reinforcement learning  control framework based on scalable graph Transformer for large-scale fuzzy Job Shop Scheduling Problems
## abstract
The Job Shop Scheduling Problem (JSSP) is a classic NP-hard problem. This paper focuses on a realistic variant of the JSSP incorporating fuzzy processing times, with the objective of minimizing the maximum completion time. We propose a Proximal Policy Optimization with Graph Transformer (GT-PPO) algorithm, which leverages Proximal Policy Optimization (PPO) as the foundational framework, to address this problem for the first time. Firstly, the intricate variability in states and actions often leads to suboptimal scheduling outcomes. To address this, we refine the representation of states and actions for improved performance. Secondly,  to overcome common limitations of Graph Neural Networks (GNNs)—such as challenges with heterogeneity, over-squashing, and capturing long-range dependencies—we, for the first time, employ Graph Transformer (GT). These transformers effectively capture both the topological relationships in fuzzy disjunctive graph models and the long-range dependencies in large-scale JSSP instances. Additionally, We also reduce the computational complexity of the GT to $O(n)$, enabling the agent to derive optimal scheduling solutions for large disjunctive graphs more efficiently, with reduced memory usage. Finally, the testing results demonstrate the strong robustness of our model across various scales of generated instances and public datasets after a single training session. Notably, on large-scale DMU and Taillard public datasets, the model exhibited exceptional robustness, further validating its effectiveness in addressing large-scale fuzzy JSSP.

## Installation
Pytorch 1.6

Gym 0.17.3
## Reproducing
To reproduce the result in the paper, first clone the whole repo:

git clone https://github.com/WenquanZ12/GT-PPO

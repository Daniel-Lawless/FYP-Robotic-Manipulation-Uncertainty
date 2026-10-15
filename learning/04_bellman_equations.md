[[_TOC_]]

### Expressing action values using the next state

$Q^{\pi} (s,a)$ is the probability-weighted average of all possible returns $G_{t}$ when starting in state $s$, taking action $a$, and following policy $\pi$ afterwards:

$$
Q^{\pi} (s, a) = E_{\pi} [G_{t} | S_{t} = s, A_{t} = a]
$$

Each return consists of the immediate reward plus the discounted remaining return:

$$
G_{t} = R_{t+1} + \gamma G_{t+1}
$$

Substituting this gives:

$$
Q^{\pi} (s, a) = E_{\pi} [R_{t+1} + \gamma G_{t+1} | S_{t} = s, A_{t} = a]
$$

We can now group the possible returns according to the next state $s'$ reached and the immediate reward $r$ received. Within each group, the immediate reward is fixed. Under the Markov property, the probability-weighted average of the remaining returns $G_{t+1}$ is $V^{\pi} (s')$, since we follow policy $\pi$ from that next state onwards.

The mean return within each group is therefore:

$$
r + \gamma V^{\pi}(s')
$$

Using the law of total expectation, we average these group means according to the probabilities of reaching their next states and receiving their rewards:

$$
Q^{\pi} (s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma V^{\pi}(s')]
$$

Equivalently, we can keep this in expectation notation:

$$
Q^{\pi} (s,a) = E_{\pi} [R_{t+1} + \gamma V^{\pi}(S_{t+1}) | S_{t} = s, A_{t} = a]
$$

Both expressions give the same probability-weighted average as the original definition using $G_{t}$. We have grouped the possible returns and averaged within each group, allowing us to express the action value using the immediate reward and the values of the possible next states.

### The Bellman expectation equation for state values

We have seen that $V^{\pi}(s)$ is the probability-weighted average of the Q-values for the available actions in state $s$:

$$
V^{\pi}(s) = \sum_{a}\pi(a\mid s)Q^{\pi}(s,a).
$$

We can now substitute our expression for each Q-value:

$$
V^{\pi}(s) =
\sum_{a}\pi(a | s)
\sum_{s',r}p(s',r | s,a)
[r+\gamma V^{\pi}(s')].
$$

This is the Bellman expectation equation for state values. It calculates the average in two stages:

1. For each possible first action $a$, average $r+\gamma V^{\pi}(s')$ over the possible next states and rewards, weighted by $p(s',r\mid s,a)$. This gives $Q^{\pi}(s,a)$.
2. Average these Q-values, weighting each one by the probability $\pi(a | s)$ of choosing that first action. This gives $V^{\pi}(s)$.

Equivalently, we can write:

$$
V^{\pi}(s)= E_{\pi}[R_{t+1}+\gamma V^{\pi}(S_{t+1}) | S_{t}=s].
$$

The expectation includes all possible first actions under policy $\pi$ and their possible next states and rewards. It gives the same probability-weighted average of returns $G_{t}$ as our original definition of $V^{\pi}(s)$, now expressed using immediate rewards and next-state values.

### Worked example

<div align="center">
  <img src="../images/bellman_transition_tree.png" alt="State transitions and action probabilities" width="1000">
</div>

Suppose we are in state $s_{1}$, where three actions are available: $a_{1}$, $a_{2}$, and $a_{3}$. Our policy selects these actions with 
probabilities $0.7$, $0.1$, and $0.2$, respectively. If we take action $a_{1}$, we reach $s_{2}$ with probability $0.3$,
$s_{3}$ with probability $0.3$, or $s_{4}$ with probability $0.4$. In this example, the reward is fixed for each transition: 
these three outcomes give immediate rewards of $1$, $2$, and $1$, respectively. The possible next states, transition probabilities,
and rewards for the other two actions are shown in the diagram. We will first calculate the probability-weighted average of the immediate
reward plus the discounted next-state value for each action, giving its Q-value. We will then average these Q-values using the policy 
probabilities to obtain $V^{\pi}(s_{1})$.

<div align="center">
  <img src="../images/bellman_action_value_charts.png" alt="Action-value distributions" width="1200">
</div>

For each action, we can visualise a probability distribution over the immediate reward plus the discounted value of the next state, $R_{t+1}+\gamma V^{\pi}(S_{t+1})$.
The Q-value is the mean of this distribution: we multiply each possible value $r+\gamma V^{\pi}(s')$ by the joint probability $p(s', r | s,a)$ of receiving that reward and
reaching that next state, then sum the results. In this example, the reward is fixed for each transition, so each bar's height is simply the probability of reaching its next 
state, given the current state and chosen action. The three graphs show these distributions for actions $a_{1}$, $a_{2}$, and $a_{3}$, with their probability-weighted 
means giving the corresponding Q-values.

<div align="center">
  <img src="../images/bellman_state_value_distribution.png" alt="Distribution over action values" width="500">
</div>

Using the law of total expectation, we can obtain $V^{\pi}(s_{1})$ by taking a probability-weighted average of the three Q-values. Each Q-value is the mean of the possible 
returns grouped by a particular first action, and the probability of selecting that group is the policy probability $\pi(a_{i}\mid s_{1})$. We therefore multiply each Q-value 
by its action probability and sum the results: $V^{\pi}(s_{1}) = 0.7Q^{\pi}(s_{1},a_{1}) + 0.1Q^{\pi}(s_{1},a_{2}) + 0.2Q^{\pi}(s_{1},a_{3})$. This weighted mean of the 
group means equals the probability-weighted average of all possible returns from $s_{1}$ under policy $\pi$.
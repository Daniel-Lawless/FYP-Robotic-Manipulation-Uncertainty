[[_TOC_]]

## Markov Reward Processes (MRP)

Markov processes only had states and transition probabilities. It could tell you "from state A you go to state B with probability 0.3", but it could not tell you if that was good or bad. To take this into account, we add rewards.

The most general way to represent rewards is a reward matrix $R$, similar to a transition matrix $T(i, j)$, we have a square matrix $R(i,j)$ witch gives a reward from transitioning from state i to state j. A reward can be positive, negative, large, or small. It is just a scalar value that tells the agent gow desirable that transition was. If the reward only depends on the state you reach, not what state you come from, then instead of $R(i, j)$ we can just use $R(x)$ i.e., just assign a reward to each state.

The second thing we're going to add is a discount factor $\gamma$ which is a single value between 0 and 1 (inclusive). It controls how much we care about future rewards. If $\gamma \approx 0$ then we care almost only about immediate reward, if $\gamma \approx 1$ then we care a lot about future rewards.

So now every transition has a reward. In MP we had $s_{0} \rightarrow s_{1} \rightarrow s_{2} \cdots$ now in an MRP we have $s_{0} \xrightarrow{r_{1}} s_{1} \xrightarrow{r_{2}} s_{2} \xrightarrow{r_{3}} \cdots$. The reason the reward index is one ahead is because it is given after we have transitioned. So $r_{1}$ is the reward for transitioning from $s_{0}$ to $s_{1}$.

### Return

We can define $G_{t}$ to be the return from time step $t$ along a particular episode, only counting rewards that come after that point. If the episode ends at time $T$:

$$
G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^{2}R_{t+3} + \cdots + \gamma^{T - t - 1}R_{T}
$$

$\gamma$ represents the shortsightedness of the agent. if $\gamma = 1$ then return is just the sum of all subsequent rewards. If $\gamma = 0$ then the return will just be the immediate reward $r_{t+1}$. In practice this value is usually between $0.9 \leq \gamma \leq 0.99$.

### Value Function

However, $G_{t}$ is not very useful in pracice since different episodes can produce different returns from the same state $s_{t}$. i.e., if we start in $s_{t}$ one episode might give a big reward (we just happen to go to states that give a higher reward), one episode mat give a small reward (we just happen to go to states that give a lower reward). $\therefore G_{t}$ is high variance, random, and depends on chance. Instead, we use the value function:

$$
V(s) = E[G_{t} | s_{t} = s]
$$

This moves from "What return did we get on a particular run" ($G_{t}$), to "what return should we expect, on average, from this state" ($V(s)$). We can think of this as a probability distribution where $G_{t}$ is on the x-axis and $p(G_{t} | s_{t} = s)$ is on the y-axis. The mean of this distribution is $V(s)$.

The true value of state $s$ is the expected return from that state:

$$
V(s) = \mathbb{E}[G_{t} \mid S_{t} = s]
$$

We estimate it by averaging $N$ sampled returns from $s$, where $G_{t}^{(i)}$ is the return from the $i$th run. Under the law of large numbers, this estimate converges to the true value as $N$ increases:

$$
\hat{V}_{N}(s) = \frac{1}{N}\sum_{i=1}^{N}G_{t}^{(i)}
\xrightarrow{N \to \infty} V(s)
$$

### Concrete Example

<div align="center">
  <img src="../images/transition_diagram.png" alt="State transition diagram" width="500">
</div>

Lets start with the case where we only care about the immediate reward, $\gamma = 0$. So we want to know the value of the states when $\gamma = 0, G_{t} = r_{t+1}$. So $V(s) = E[r_{t+1} | s_{t} = s]$. To find the mean of a discrete distribution we can use $\sum^{n}_{i=1}(\text{probability} \times \text{reward})$.

$V(\text{chat}) = 0.5(-1) + 0.2(1) + 0.3(2) = 0.3$

$V(\text{computer}) = 0.5(5) + 0.2(2) + 0.2(1) + 0.1(-3) = 2.8$

$V(\text{coffee}) = 0.7(2) + 0.2(3) + 0.1(1) = 2.1$

$V(\text{home}) = 0.6(1) + 0.4(1) = 1.0$

If we only care about immediate rewards ($\gamma=0$), Computer is the most valuable state to be in because it has the highest expected immediate reward. In this example, setting ($\gamma=1$) would make the expected return infinite from every starting state: the process continues indefinitely, and every state has an expected immediate reward of at least 0.3, so the expected accumulated reward grows without bound. Choosing ($0 \lt \gamma \lt 1$) gives progressively less weight to rewards further into the future and, because the rewards are bounded, keeps the infinite discounted return finite.

The probability of a given $G_{t}$ is the product of the transition probabilities across that path. For example, using the diagram, the return of the path Chat $\rightarrow$ Computer $\rightarrow$ Home is $G_{t} = 2 + \gamma (2)$, its probability is $0.3 \times 0.2 = 0.06$. We can think of this as one bar chart on our distribution for V(s). An important distinction to make is that different paths can have the same return. To find the probability of a particular return value, we add all the probabilities of all paths producing it.
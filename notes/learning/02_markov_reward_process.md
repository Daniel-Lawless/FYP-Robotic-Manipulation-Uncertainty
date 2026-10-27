[[_TOC_]]

## Markov Reward Processes (MRP)

Markov processes only had states and transition probabilities. It could tell you "from state A you go to state B with probability 0.3", but it could not tell you if that was good or bad. To take this into account, we add rewards.

The most general way to represent rewards is a reward matrix $R$, similar to a transition matrix $T(i, j)$, we have a square matrix $R(i,j)$ which gives a reward from transitioning from state i to state j. A reward can be positive, negative, large, or small. It is just a scalar value that tells us how desirable that transition was. If the reward only depends on the state you reach, not what state you come from, then instead of $R(i, j)$ we can just use $R(x)$ i.e., just assign a reward to each state.

The second thing we're going to add is a discount factor $\gamma$ which is a single value between 0 and 1 (inclusive). It controls how much we care about future rewards. If $\gamma \approx 0$ then we care almost only about immediate reward, if $\gamma \approx 1$ then we care a lot about future rewards.

So now every transition has a reward. In MP we had $s_0 \rightarrow s_1 \rightarrow s_2 \cdots$ now in an MRP we have $s_0 \xrightarrow{r_1} s_1 \xrightarrow{r_2} s_2 \xrightarrow{r_3} \cdots$. The reason the reward index is one ahead is because it is given after we have transitioned. So $r_1$ is the reward for transitioning from $s_0$ to $s_1$.

### Return

We can define $G_t$ to be the return from time step $t$ along a particular episode, only counting rewards that come after that point. If the episode ends at time $T$:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^{2}R_{t+3} + \cdots + \gamma^{T-t-1}R_T
$$

$\gamma$ represents the shortsightedness of the agent. If $\gamma = 1$ then return is just the sum of all subsequent rewards. If $\gamma = 0$ then the return will just be the immediate reward $R_{t+1}$. In practice this value is often between $0.9 \leq \gamma \leq 0.99$.

### Value Function

The return $G_t$ is a random variable, meaning different episodes can produce different returns from the same state $s_t$. For example, one episode might give a big return (we happen to transition to states that give higher rewards), while another episode might give a small return (we happen to transition to states that give lower rewards).

Therefore, a single sampled return does not necessarily tell us how valuable a state is on average. Instead, we use the state-value function:

$$
v(s) = \mathbb{E}[G_t \mid S_t = s]
$$

This moves from "What return did we get on a particular run" ($G_t$), to "What return should we expect, on average, from this state" ($v(s)$). We can think of this as a probability distribution where $G_t$ is on the x-axis and $p(G_t \mid S_t = s)$ is on the y-axis. The mean of this distribution is $v(s)$.

The **true value** of state $s$ is the expected return from that state:

$$
v(s) = \mathbb{E}[G_t \mid S_t = s]
$$

We can **estimate** this value by averaging $N$ sampled returns from $s$, where $G_t^{(i)}$ is the return from the $i$th sample.

We use uppercase $V_N(s)$ to distinguish our estimate from the true value $v(s)$.

Under the law of large numbers, this estimate converges to the true value as $N$ increases, assuming the returns are sampled appropriately:

$$
V_N(s) = \frac{1}{N}\sum_{i=1}^{N}G_t^{(i)}
\xrightarrow{N \to \infty} v(s)
$$

### Concrete Example

<div align="center">
  <img src="../images/transition_diagram.png" alt="State transition diagram" width="500">
</div>

Let's start with the case where we only care about the immediate reward, $\gamma = 0$. So we want to know the true values of the states when $\gamma = 0$, meaning $G_t = R_{t+1}$. Therefore:

$$
v(s) = \mathbb{E}[R_{t+1} \mid S_t = s]
$$

To find the mean of a discrete distribution we can use $\sum_{i=1}^{n}(\text{probability} \times \text{reward})$.

$v(\text{chat}) = 0.5(-1) + 0.2(1) + 0.3(2) = 0.3$

$v(\text{computer}) = 0.5(5) + 0.2(2) + 0.2(1) + 0.1(-3) = 2.8$

$v(\text{coffee}) = 0.7(2) + 0.2(3) + 0.1(1) = 2.1$

$v(\text{home}) = 0.6(1) + 0.4(1) = 1.0$

If we only care about immediate rewards ($\gamma=0$), Computer is the most valuable state to be in because it has the highest expected immediate reward.

In this example, setting $\gamma=1$ would make the expected return infinite from every starting state: the process continues indefinitely, and every state has an expected immediate reward of at least 0.3, so the expected accumulated reward grows without bound. Choosing $0 < \gamma < 1$ gives progressively less weight to rewards further into the future and, because the rewards are bounded, keeps the infinite discounted return finite.

The probability of a particular path is the product of the transition probabilities along that path. For example, using the diagram, the two-transition path Chat $\rightarrow$ Computer $\rightarrow$ Home has a discounted reward sum of $2 + \gamma(2)$, and its probability is $0.3 \times 0.2 = 0.06$.

If the episode terminates at Home, this sum is the complete return $G_t$. Otherwise, it is only the first two terms of the return.

Different paths can produce the same return. To find the probability of a particular return value, we add the probabilities of all complete paths producing it.
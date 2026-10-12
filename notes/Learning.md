# This file will include the material that I have learned or currently learning.

### Markov decision process:
Before I move onto Markov decision processes (MDP), I'll start with the simplest case of the Markov family, the Markov process. Leter, I will extend this with rewards whcih will turn it into a markov reward process (MRP), then I will add actions which will lead to the MDP.

**The markov Process (MP) also called the Markov chain**:

This is the simplest stochasitc model of a process that changes overtime. The key constraints are:

- You cannot contrl it (no action)
- You can only observe it
- The future depends only on the present, not the past (Markov property)

Imagine we have a system infront of us that we can only observe. What we observe is called states, and the system switches between states. We cannot influence the system, only watch the state change. All possible states for a system form the systems state space, and for an MP we require this set to be finite. Our observations form a sequence of states or a chain, hence why it is sometimes called a markov chain.

*Example*:

Lets say we are watching the weather. We can observe the current day as rainy or sunny, which makes up our state space. A sequence of observations overtime form a chain of states, such as ["rainy", "sunny", "sunny", "rainy", "sunny", ..., "rainy"]. This is called history. To call this system an MP it needs to fulful the markov property. $P(s_{t+1} | s_{t}, s_{t-1},...,s_{0}) = P(s_{t+1} | s_{t})$. So the probability of state $s_{t+1}$ only depends on state $s_{t}$. To use our example, the probability of whether or not tomorrow is sunny, $s_{t+1}$, only depends on todays weather, $s_{t}$.

This means every state is self-contained, meaining a state must contain all the information needed to predict the next state. Since this hols for all states, it allows us to predict all future states. i.e., $s_{t} \rightarrow s_{t+1} \rightarrow s_{t+2} \rightarrow \cdots$

So, the markov property states that the current state $s_{t}$ contains all the information needed to predict the distribution (not a single deterministic outcome) over the next state $s_{t+1}$. Here The distribution are all the values $s_{t+1}$ could take and the probability of each one.

**Transition matrix**:

We can capture transition probabilities with a transition matrix, which is an N x N matrix, Where N is the number of states in our system. Each row represents a conditional probability distribution over the next state, given the current state, so its entries sum to 1. For discrete states, we can visualise a row as a bar chart, with possible next states on the x-axis and their probabilities on the y-axis.

In practice, we rarely know the transition matrix, instead what we observe is a sequence of states overtime. These sequences are called trajectories.

## Markov reward processes
Markov processes only had states and transition probabilities. It could tell you "from state A you go to state B with probability 0.3", but it could not tell you if that was good or bad. To take this into account, we add rewards.

The most general way to represent rewards is a reward matrix $R$, similar to a transition matrix $T(i, j)$, we have a square matrix $R(i,j)$ witch gives a reward from transitiong from state i to state j. A reward can be positive, negative, large, or small. It is just a scalar value that tells the agent gow desirable that transition was. If the reward only depends on the state you reach, not what state you come from, then instead of $R(i, j)$ we can just use $R(x)$ i.e., just assign a reward to each state.

The second thing we're going to add is a discount factor $\gamma$ which is a single value between 0 and 1 (inclusive). It controls how much we care about future rewards. If $\gamma \approx 0$ then we care almost only about immediate reward, if $\gamma \approx 1$ then we care a lot about future rewards.

So now every transtion has a reward. In MP we had $s_{0} \rightarrow s_{1} \rightarrow s_{2} \cdots$ now in an MRP we have $s_0 \xrightarrow{r_1} s_1 \xrightarrow{r_2} s_2 \xrightarrow{r_3} \cdots$. The reason the reward index is one ahead is because it is given after we have transitioned. So $r_{1}$ is the reward for transitioning from $s_{0}$ to $s_{1}$. 

**Return:**

We can define $G_{t}$ to be the return from time step $t$ along a particular episode, only counting rewards that come after that point. If the episode ends at time $T$:

$$
G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^{2}R_{t+3} + \cdots + \gamma^{T - t - 1}R_{T}
$$

$\gamma$ represents the shortsightedness of the agent. if $\gamma = 1$ then return is just the sum of all subsequent rewards. If $\gamma = 0$ then the return will just be the immediate reward $r_{t+1}$. In practice this value is usually between $0.9 \leq \gamma \leq 0.99$.

However, $G_{t}$ is not very useful in pracice since different episodes can produce different returns from the same state $s_{t}$. i.e., if we start in s_{t} one episode might give a big reward (we just happen to go to states that give a higher reward), one episode mat give a small reward (we just happen to go to states that give a lower reward). $\therefore G_{t}$ is high variance, random, and depends on chance. Instead, we use the value function:

$$
V(s) = E[G_{t} | s_{t} = s]
$$

This moves from "What return did we get on a particular run" ($G_{t}$), to "what return should we expect, on average, from this state" ($V(s)$). We can think of this as a probability distribution where $G_{t}$ is on the x-axis and $p(G_{t} | s_{t} = s)$ is on the y-axis. The mean of this distribution is $V(s)$.

The true value of state $s$ is the expected return from that state:

$$
V(s) = \mathbb{E}[G_t \mid S_t = s]
$$

We estimate it by averaging $N$ sampled returns from $s$, where $G_t^{(i)}$ is the return from the $i$th run. Under the law of large numbers, this estimate converges to the true value as $N$ increases:

$$
\hat{V}_N(s) = \frac{1}{N}\sum_{i=1}^{N}G_t^{(i)}
\xrightarrow{N \to \infty} V(s)
$$

**Concrete example**:

![State transition diagram](../images/transition_diagram.png)



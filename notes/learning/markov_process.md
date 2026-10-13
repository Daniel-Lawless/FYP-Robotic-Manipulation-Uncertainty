[[_TOC_]]

## Markov Process (MP):

Before I move onto Markov decision processes, MDP, I'll start with the simplest case of the Markov family, the Markov process. Later, I will extend this with rewards which will turn it into a markov reward process, MRP, then I will add actions which will lead to the MDP.

**The Markov Process also called the Markov Chain:**

This is the simplest stochasitc model of a process that changes overtime. The key constraints are:

- You cannot contrl it (no action)
- You can only observe it
- The future depends only on the present, not the past (Markov property)

Imagine we have a system infront of us that we can only observe. What we observe is called states, and the system switches between states. We cannot influence the system, only watch the state change. All possible states for a system form the systems state space, and for an MP we require this set to be finite. Our observations form a sequence of states or a chain, hence why it is sometimes called a markov chain.

### Concrete Example

Lets say we are watching the weather. We can observe the current day as rainy or sunny, which makes up our state space. A sequence of observations overtime form a chain of states, such as ["rainy", "sunny", "sunny", "rainy", "sunny", ..., "rainy"]. This is called history. To call this system an MP it needs to fulful the markov property. $P(s_{t+1} | s_{t}, s_{t-1},...,s_{0}) = P(s_{t+1} | s_{t})$. So the probability of state $s_{t+1}$ only depends on state $s_{t}$. To use our example, the probability of whether or not tomorrow is sunny, $s_{t+1}$, only depends on todays weather, $s_{t}$.

This means every state is self-contained, meaining a state must contain all the information needed to predict the next state. Since this hols for all states, it allows us to predict all future states. i.e., $s_{t} \rightarrow s_{t+1} \rightarrow s_{t+2} \rightarrow \cdots$

So, the markov property states that the current state $s_{t}$ contains all the information needed to predict the distribution (not a single deterministic outcome) over the next state $s_{t+1}$. Here The distribution are all the values $s_{t+1}$ could take and the probability of each one.

### Transition Matrix:

We can capture transition probabilities with a transition matrix, which is an N x N matrix, Where N is the number of states in our system. Each row represents a conditional probability distribution over the next state, given the current state, so its entries sum to 1. For discrete states, we can visualise a row as a bar chart, with possible next states on the x-axis and their probabilities on the y-axis.

In practice, we rarely know the transition matrix, instead what we observe is a sequence of states overtime. These sequences are called trajectories.
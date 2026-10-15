[[_TOC_]]

## Markov Decision Process (MDP)

To extend MRP to MDP we add a set of actions $A$, which has to be finite. This makes our agents action space. For an MDP, the agent no longer passively observes state transitions, but can suggest an action to take at every state transition. This now means the agent can influence outcomes, which is a useful ability.

### Policy

The simple definition of a policy is that it is just a rule, or set of rules, that tell our agent what actions to take, usually written as $\pi(a | s)$. The main objective in reinforcement learning is to maximise the expected return. Formally, polity is defined as the probability distribution over actions for every possible state:

$$
\pi(a | s) = p[A_{t} = a | S_{t} = s]
$$

So for a given state, s, the policy gives a distribution over the actions we can take from that state, where actions are on the x-axis, and probability (denisty if a is continuous) is on the y-axis.

### Value functions

#### State Value Function

$V^{\pi}(s)$ is the probability-weighted average of all possible returns $G_{t}$ when starting in state $s$ and following policy $\pi$. This includes all possible first actions selected by the policy and all possible outcomes afterwards.

$$
V^{\pi} = E_{\pi}[G_{t} | S_{t} = s]
$$

#### Action Value Function

$Q^{\pi}(s,a)$ is the probability-weighted average of all possible returns $G_{t}$ when starting in state $s$, taking action $a$, and following policy $\pi$ afterwards. The first action is fixed, but we still average over all possible outcomes and subsequent actions.

$$
Q^{\pi}(s,a) = E_{\pi}[G_{t} | S_{t} = s, A_{t} = a]
$$

The distinction is that $V^{\pi}(s)$ averages the possible returns $G_{t}$ across all possible first actions under policy $\pi$, whereas $Q^{\pi}(s,a)$ averages the possible returns $G_{t}$ when the first action is fixed to $a$. Both averages account for all possible outcomes and subsequent actions, weighted by their probabilities.

#### Connecting state values and action values

Starting in state $s$, we can group the possible returns $G_{t}$ according to the first action taken. For each action $a$, the probability-weighted average of the returns in that group is $Q^{\pi}(s,a)$.

To obtain $V^{\pi}(s)$, we average these Q-values, weighting each one by the probability $\pi (a | s)$ of choosing that first action:

$$
V^{\pi} (s) = \sum_{a} \pi(a | s) Q^{\pi}(s,a)
$$

This is the law of total expectation: averaging the possible returns within each first-action group, then weighting those group means by the probabilities of choosing the actions, gives the same result as averaging all possible returns from $s$ directly.
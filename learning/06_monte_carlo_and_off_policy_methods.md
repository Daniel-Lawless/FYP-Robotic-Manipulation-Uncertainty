[[_TOC_]]

### Monte Carlo Methods:

Monte Carlo methods can be used when we do not know the environement dynamics, specifically $p(s', r | s, a)$. We can think of this term as the transition probabilities between each state, along with what reward we will get for that transition.

Instead, we run some policy $\pi$ through the MDP and get samples of trajectories:

$$
S_{0}, A_{0}, R_{1}, S_{1}, A_{1}, R_{2}, S_{2}, A_{2} \cdots , R_{T}, S_{T}
$$

Monte Carlo using averages over trajectories to approximate expected values and GPI to obtain politices to approximate the optimal policy. MC does not approximate $p(s', r | s, a)$.

#### Model-Free vs Model-Based methods

We can think of a model as anything the agent uses to predict how the environement will respond to its action. An example of a model would be $p(s', r | s, a)$, since we can use it to get an idea of the state we will land in and the reward we will get for some action.

- **Model-Based methods:** These are methods that use a model to plan the actions it takes.

- **Model-Free methods:** These are methods that do not use a model. They instead learn associations between actions and returns

Monte Carlo methods are model free, they do not use a model to predict how the environement will respond to actions. Because of this, we need to estimate action values instead of state values. To see this, lets say we estimated $v_{*}(s)$ with $V(s)$. To find the max-value action, we still have to form an estimate of q_{*}(s,a):

$$
Q(s,a) = \sum_{s' \in S, r \in R} p(s', r | s, a) [r + \gamma V(s')]
$$

But because Monte Carlo is model-free, we do not have $p(s', r | s, a)$, meaning we would have to estimate $q_{*}(s, a)$ anyway, so we may as well do that directly.


#### Monte Carlo Evaluation

The goal of Monte Carlo evaluation is: given sample trajectories generated under a policy $\pi$, estimate $q_{\pi}$, i.e. estimate the action value for state-action pairs.

When doing Monte Carlo evaluation, we can express estimation of $q_{\pi}$ as estimation of a state-value function in a new process.

To see this, start from the usual state-value definition:

$$
v_{\pi}(s) = E_{\pi}[G_{t} | S_{t} = s]
$$

Now define the state of a new problem to be a state-action pair:

$$
S^{new}_{t} = (S_{t}, A_{t})
$$

Then the value of one of these new states is:

$$
v^{new}_{\pi}((s,a)) = E_{\pi}[G_{t} | S^{new}_{t} = (s,a)]
$$

But saying $S^{new}_{t} = (s,a)$ means exactly that, in the original problem, we are in state $s$ and take action $a$. So this becomes:

$$
v^{new}_{\pi}((s,a)) = E_{\pi}[G_{t} | S_{t} = s, A_{t} = a]
$$

And this is precisely the definition of the action-value function:

$$
q_{\pi}(s,a) = E_{\pi}[G_{t} | S_{t} = s, A_{t} = a]
$$

So we get:

$$
v^{new}_{\pi}((s,a)) = q_{\pi}(s,a)
$$

Therefore, estimating $v^{new}_{\pi}((s,a))$ in this new problem is the same as estimating $q_{\pi}(s,a)$ in the original problem.

Since we have fixed the policy $\pi$, and actions are now encoded in the state, this new process is a Markov Reward Process (MRP), i.e. a process with no explicit action choices. Its trajectories look like:

$$
S_{0}, R_{1}, S_{1}, R_{2}, S_{2}, \cdots, R_{T}, S_{T}
$$

So our data consists of $M$ trajectory samples from an MRP. We can write one trajectory sample as:

$$
s^{m}_{0}, r^{m}_{1}, s^{m}_{1}, \cdots, s^{m}_{T_{m}}, \quad m = 1, \ldots, M
$$

Where $M$ is the number of trajectories. The superscript $m$ tells us which trajectory we are in, and the subscript tells us the position within that trajectory.

The idea of Monte Carlo is to use averages to approximate the value function. For ordinary state values this gives:

$$
E_{\pi}[G_{t} | S_{t} = s] \approx \frac{1}{C(s)} \sum^{M}_{m=1} \sum^{T_{m}-1}_{\tau=0} \mathbb{I}[s^{m}_{\tau}=s]g^{m}_{\tau}
$$

Since we have defined states to be state-action pairs, we get the Monte Carlo estimate for the action-value function:

$$
q_{\pi}(s,a) \approx Q_{\pi}(s,a) = \frac{1}{C(s,a)} \sum^{M}_{m=1} \sum^{T_{m}-1}_{\tau=0} \mathbb{I}[(s^{m}_{\tau},a^{m}_{\tau})=(s,a)]g^{m}_{\tau}
$$

Here, $C(s,a)$ is the total number of times the state-action pair $(s,a)$ appears across all trajectories, and $g^{m}_{\tau}$ is the return observed from time step $\tau$ onward in trajectory $m$.

**Example:**

<div align="center">
  <img src="../images/trajectory_diagram.png" alt="Monte Carlo trajectories" width="850">
</div>

Let us estimate the value of the state-action pair $(s_{1},a_{1})$.

$$
q_{\pi}(s_{1},a_{1}) \approx Q_{\pi}(s_{1},a_{1}) = \frac{1}{C(s_{1},a_{1})} \sum^{M}_{m=1} \sum^{T_{m}-1}_{\tau=0} \mathbb{I}[(s^{m}_{\tau},a^{m}_{\tau})=(s_{1},a_{1})]g^{m}_{\tau}
$$

This equation tells us to go through each trajectory one by one, and every time we encounter the state-action pair $(s_{1},a_{1})$, we add the return from that time step.

For this example, we assume $\gamma = 1$, so the return is just the sum of all future rewards.

For the first trajectory, we see $(s_{1},a_{1})$ twice: once at time step $\tau=0$, and once at time step $\tau=2$. So the total contribution from trajectory 1 is:

$$
g^{1}_{0} + g^{1}_{2} = (2 + 1 + 3) + (3) = 9
$$

For the second trajectory, we see $(s_{1},a_{1})$ once, at time step $\tau=1$. The return from that point is:

$$
g^{2}_{1} = 4 + 2 + 1 = 7
$$

For the third trajectory, we see $(s_{1},a_{1})$ once, at time step $\tau=0$. The return from that point is:

$$
g^{3}_{0} = 0 + 2 + 1 = 3
$$

So the total contribution is:

$$
9 + 7 + 3 = 19
$$

The state-action pair $(s_{1},a_{1})$ appears a total of 4 times across all trajectories, so:

$$
C(s_{1},a_{1}) = 4
$$

Therefore, our Monte Carlo estimate is:

$$
q_{\pi}(s_{1},a_{1}) \approx Q_{\pi}(s_{1},a_{1}) = \frac{1}{4}(9 + 7 + 3) = \frac{19}{4} = 4.75
$$

So our estimate of the true action value $q_{\pi}(s_{1},a_{1})$ is:

$$
Q_{\pi}(s_{1},a_{1}) = 4.75
$$

This is an example of every-visit Monte Carlo, because we include the return from every occurrence of the state-action pair $(s_{1},a_{1})$, even if it appears multiple times in the same trajectory.

However, there is one limitation with our previous approach. To calculate the estimate using the batch formula, we would need to collect the returns from our sampled trajectories and then compute their average. Instead, we can calculate this average incrementally, updating our estimate whenever a new return becomes available.

We can do this using the formula:

$$
Q_{\pi}(s,a) \leftarrow Q_{\pi}(s,a) + \frac{1}{C(s,a)}(g_{\tau}^{m} - Q_{\pi}(s,a))
$$

This produces exactly the same result as calculating the average directly. However, instead of storing all the previous returns for a state-action pair, we only need to maintain our current estimate $Q_{\pi}(s,a)$ and the number of returns observed so far, $C(s,a)$.

To demonstrate this, let's return to our previous example. This time, rather than having all three trajectories available from the beginning, we collect them one at a time and update our estimate as each trajectory finishes.

Assume we initialise our estimate to:

$$
Q_{\pi}(s_{1},a_{1}) = 0
$$

We begin by running the first trajectory. Once it finishes, we can calculate the returns from each occurrence of $(s_{1},a_{1})$.

The first occurrence is at time step $\tau=0$, giving us $g^{1}_{0}=6$. Since this is our first observed return, $C(s_{1},a_{1})=1$, so we update our estimate:

$$
Q_{\pi}(s_{1},a_{1}) \leftarrow 0 + \frac{1}{1}(6-0) = 6
$$

We then take the second return from the same trajectory, $g^{1}_{2}=3$. We have now observed two returns, so $C(s_{1},a_{1})=2$:

$$
Q_{\pi}(s_{1},a_{1}) \leftarrow 6 + \frac{1}{2}(3-6) = 4.5
$$

Next, we run the second trajectory. Once it finishes, we obtain the return $g^{2}_{1}=7$ from its occurrence of $(s_{1},a_{1})$. This is our third observed return, so:

$$
Q_{\pi}(s_{1},a_{1}) \leftarrow 4.5 + \frac{1}{3}(7-4.5) = \frac{16}{3} \approx 5.333
$$

Finally, we run the third trajectory. This gives us the return $g^{3}_{0}=3$, which is our fourth observed return. We update our estimate one last time:

$$
Q_{\pi}(s_{1},a_{1}) \leftarrow \frac{16}{3} + \frac{1}{4}\left(3-\frac{16}{3}\right) = \frac{19}{4} = 4.75
$$

We have therefore obtained exactly the same estimate as before:

$$
Q_{\pi}(s_{1},a_{1}) = 4.75
$$

The difference is that we calculated the average incrementally, without needing to store all the previous returns or wait until all $M$ trajectories had been collected.

As we collect more samples, $C(s,a)$ increases, meaning the term $\frac{1}{C(s,a)}$ becomes smaller. This means each new return has a smaller influence on our current estimate, making it progressively more stable as we observe more returns. Although returns from earlier trajectories can cause larger changes to our estimate, they do not have more weight in the final result. Since we are computing the average incrementally, every observed return contributes equally to our final estimate.

One important detail is that we still need each trajectory to finish before calculating its returns, since the return $g^{m}_{\tau}$ depends on the rewards received from time step $\tau$ until the end of that trajectory. However, once these returns become available, we can immediately use them to update our estimate.

#### Constant-$\alpha$ Monte Carlo

Previously, we used $\frac{1}{C(s,a)}$ as our step size when updating our estimate. However, we can instead replace this with a constant step size $\alpha$, giving us:

$$
Q_{\pi}(s,a) \leftarrow Q_{\pi}(s,a) + \alpha(g^{m}_{\tau} - Q_{\pi}(s,a))
$$

Here, $\alpha$ is our learning rate, typically between 0 and 1, which determines how much we move our current estimate towards the newly observed return. For example, if $\alpha = 0.1$, we move 10% of the distance between our current estimate and the new return.

The key difference is how we weight our observations. With our previous incremental averaging method, every observed return has equal weight in the current average. Although the step size $\frac{1}{C(s,a)}$ gets smaller as we collect more samples, the contributions of all previous returns are also reduced equally. Therefore, older returns are not given more weight than newer ones.

With constant-$\alpha$ Monte Carlo, however, newer returns receive more weight than older ones. To understand why, we can rearrange our update rule:

$$
Q_{\pi}(s,a) \leftarrow (1-\alpha)Q_{\pi}(s,a) + \alpha g^{m}_{\tau}
$$

This shows that whenever we update our estimate, we retain a fraction $(1-\alpha)$ of our previous estimate and give the new return a weight of $\alpha$.

However, our previous estimate already contains contributions from earlier returns. This means that every time we perform an update, the influence of all previous returns is multiplied by $(1-\alpha)$ again.

For example, if $\alpha = 0.5$, the newest return receives a weight of 0.5, the previous return has a weight of 0.25, the one before that has a weight of 0.125, and so on.

More generally, after observing $n$ returns $g_{1}, g_{2}, \ldots, g_{n}$, we can express our estimate as:

$$
Q_{n} = (1-\alpha)^{n}Q_{0} + \sum^{n}_{i=1}\alpha(1-\alpha)^{n-i}g_{i}
$$

Here, $Q_{0}$ is our initial estimate, and $g_{i}$ is the $i$-th observed return for our state-action pair.

Notice that the weight of each return is:

$$
\alpha(1-\alpha)^{n-i}
$$

As an observation gets older, $n-i$ becomes larger, meaning its weight decreases exponentially.

Therefore, unlike ordinary incremental averaging, constant-$\alpha$ Monte Carlo gives greater weight to more recent observations. This can be particularly useful in non-stationary environments, where the expected returns may change over time, since our estimate can adapt to newer observations instead of being heavily influenced by old ones.

One important consequence is that constant-$\alpha$ Monte Carlo no longer computes the exact arithmetic average of all observed returns. Instead, it maintains a recency-weighted estimate.
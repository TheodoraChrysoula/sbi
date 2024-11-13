# Approximate Bayesian Computation (ABC) Algorithm

Bayes' theorem relates the conditional probability (or density) of a particular parameter value $\theta$ given the observations $y_{obs}$ to the likelihood of the observation given the paramteter $\theta$. 
The formula of the Bayes' theorem is:
$$
P(\theta | y) = \frac{P(y | \theta) P(\theta)}{P(y)}
$$

where $P(\theta | y)$ is the posterior probability, $p(\theta)$ is the prior belief we have on θ and $P(y | \theta)$ is the likelihhod function.

There are cases where the likelihood function is untractable, due to the existence of latent variables.

The ABC Rejection Algorithm is a common method where one can approximate the likelihood function by simulations, the outcomes of which are compared with the observed data. More specifically, with the ABC rejection algorithm, a set of parameter points is first sampled from the prior distribution. Given a sampled parameter point θ, a dataset $\hat{D}$ is then simulated under the statistical model $M$ specified by $\theta$. if the generated $\hat{D}$ is too different from the observed data $D$, the sampled parameter value is discarded. In precise terms, $\hat{D}$ is accepted with tolerance $\varepsilon \geq 0$ if:

$$
d(\hat{D}, D) \leq \varepsilon
$$

**Algorithm for ABC:**
1. Draw \( \theta \) from the prior.
2. Generate the simulated data \(g(\theta, x) = \mathcal{N}(\theta_1 x_1 + \theta_2 x_2, \sigma^2) \).
3. Calculate the distance \( d(f(\theta, x), y) \).
4. Accept the samples if \( d(f(\theta, x)), < \varepsilon \) (threshold).
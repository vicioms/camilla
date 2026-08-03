We consider $N$ agents with wealths $W_i(t)$ and prior market expectations $q_i$. The total capitals on the outcomes are $C_0(t)$ and $C_1(t)$. The implied probability is $p(t) = C_0(t)/(C_0(t)+C_1(t))$. We denote the total capital by $C(t)=C_0(t)+C_1(t)$.

We assume that an agent buys $\Delta_i(t) q_i$ of $0$ and $\Delta_i(t) (1-q_i)$ of $1$. Thus the changes in the capitals are:

$$ C_0(t+dt) = C_0(t) + \Delta_i(t) q_i $$

$$ C_1(t+dt) = C_1(t) + \Delta_i(t) (1-q_i) $$

Therefore,

$$p(t+dt) =  \frac{C_0(t) + q_i \Delta_i(t)}{C_0(t) + C_1(t) + \Delta_i(t)}  $$

Assuming $\Delta_i(t)$ is small:

$$p(t+dt) = p(t) \frac{1+\frac{q_i \Delta_i(t)}{C_0(t)}}{1 + \frac{\Delta_i(t)}{C(t)}} $$

$$p(t+dt) \approx p(t) \left(1+\frac{q_i \Delta_i(t)}{C_0(t)}\right) \left(1-  \frac{\Delta_i(t)}{C(t)}\right) $$

Using $C_0(t)=p(t)C(t)$ and neglecting terms of order $\Delta_i(t)^2$, we obtain

$$p(t+dt) \approx p(t)+\frac{(q_i-p(t))\Delta_i(t)}{C(t)} $$

We assume that

$$\Delta_i(t) = \gamma_i W_i(t) dt$$

and therefore

$$p(t+dt)  \approx p(t)+\frac{(q_i-p(t))\gamma_i W_i(t)}{C(t)}dt $$

The wealth of the agent drops by:

$$W_i(t+dt) = W_i(t) - \gamma_i W_i(t) dt$$

However, we now consider Poissonian investments with rate $1$. Thus, for all the agents,

$$W_i(t+dt) = W_i(t) - \gamma_i W_i(t) dB_i$$

where $dB_i$ is the increment of a Poisson process with rate $1$.

The capitals invested in the two outcomes evolve according to

$$ C_0(t+dt) = C_0(t) + \sum_{i=1}^N q_i \gamma_i W_i(t) dB_i $$

$$ C_1(t+dt) = C_1(t) + \sum_{i=1}^N (1-q_i) \gamma_i W_i(t) dB_i $$

Therefore, the total capital evolves according to

$$ C(t+dt) = C(t) + \sum_{i=1}^N \gamma_i W_i(t) dB_i $$

The implied probability becomes

$$p(t+dt) = \frac{C_0(t)+\sum_{i=1}^N q_i\gamma_i W_i(t)dB_i}{C(t)+\sum_{i=1}^N \gamma_i W_i(t)dB_i} $$

Assuming that the invested amounts are small compared with $C(t)$, we obtain

$$p(t+dt) \approx p(t)+\frac{1}{C(t)}\sum_{i=1}^N (q_i-p(t))\gamma_i W_i(t)dB_i $$

or equivalently,

$$p(t+dt)-p(t) \approx \frac{1}{C(t)}\sum_{i=1}^N (q_i-p(t))\gamma_i W_i(t)dB_i $$

where

$$\mathbb{E}[dB_i]=dt$$

for a Poisson process with rate $1$. Using a Gaussian approximation:

$$p(t+dt)-p(t) \approx \frac{1}{C(t)}\sum_{i=1}^N (q_i-p(t))\gamma_i W_i(t) [dt + dZ_i] $$

$$ C(t+dt) = C(t) + \sum_{i=1}^N \gamma_i W_i(t) [dt + dZ_i] $$

$$W_i(t+dt) = W_i(t) - \gamma_i W_i(t) [dt+dZ_i]$$

Thus:

$$ dp(t) = \frac{1}{C(t)} \sum_{i=1}^N (q_i-p(t)) \gamma_i W_i(t) dt + \frac{1}{C(t)} \sum_{i=1}^N (q_i-p(t)) \gamma_i W_i(t) dZ_i(t)  $$


$$ dC(t) = \sum_{i=1}^N \gamma_i W_i(t) dt  +  \sum_{i=1}^N \gamma_i W_i(t) dZ_i(t) $$

$$dW_i(t) = - \gamma_i W_i(t) dt - \gamma_i W_i(t) dZ_i(t)$$


Now:

$dx_i(t) = d \log W_i(t) - d\log C(t) $

$$ d \log W_i(t) = \frac{dW_i}{W_i} - \frac{1}{2} \frac{(dW_i(t))^2}{W_i^2(t)} $$

$$ d \log W_i(t) = -\left(\gamma_i dt + \frac{1}{2} \gamma_i^2 \right) dt -\gamma_i dZ_i(t) $$

$$ d \log C(t) = \frac{dC(t)}{C(t)} - \frac{1}{2} \frac{(dC(t))^2}{C(t)^2}  $$

$$ d \log C(t) =  \sum_{i=1}^N \gamma_i \frac{W_i(t)}{C(t)} dt  +  \sum_{i=1}^N \gamma_i \frac{W_i(t)}{C(t)} dZ_i(t) - \frac{1}{2C(t)^2} \left[ \sum_{i=1}^N \gamma_i^2 W_i(t)^2 dt \right]  $$

$$ d \log C(t) =  \sum_{i=1}^N \left( \gamma_i w_i(t) - \frac{\gamma_i^2}{2} w_i^2(t) \right) +   \sum_{i=1}^N \gamma_i w_i(t) dZ_i(t)  $$


$$ dx_i =  -\left(\gamma_i dt + \frac{1}{2} \gamma_i^2 \right) dt -\gamma_i dZ_i(t)  - \sum_{j=1}^N \left( \gamma_j e^{x_j(t)} - \frac{\gamma_j^2}{2}e^{2x_j(t)} \right) -   \sum_{j=1}^N \gamma_i e^{x_j(t)} dZ_j(t)$$

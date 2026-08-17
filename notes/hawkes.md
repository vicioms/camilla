The rate is:

$$ \lambda_{t+dt} = 
\begin{cases}
\lambda_t + \beta(\mu -   \lambda_t) dt & \textrm{w.p.} \, 1-\lambda_t dt \\
\lambda_t + \alpha + \beta(\mu -   \lambda_t) dt & \textrm{w.p.} \, \lambda_t dt
\end{cases}$$

$$ d\lambda_t = \beta (\mu - \lambda_t) dt + \alpha dN_t $$

$$ f(\lambda_{t+dt}) = (1-\lambda_t dt) f(\lambda_t + \beta (\mu - \lambda_t) dt ) +  \lambda_t dt f(\lambda_t + \alpha  \beta (\mu - \lambda_t) dt )  $$

$$\mathcal{L} = \beta (\mu-\lambda) \partial_\lambda + \lambda (T_{\alpha}-1)  $$

$$ \mathcal{L}^\dagger = - \beta \partial_\lambda \left((\mu-\lambda)  .\right) - \lambda . + (\lambda-\alpha) T_{-\alpha} . $$

$$\partial_t p(\lambda,t)
 = - \beta \partial_\lambda \left((\mu-\lambda) p(\lambda, t)\right) + (\lambda-\alpha) p(\lambda-\alpha,t) - \lambda p(\lambda,t)$$


 Looking at the backward equation in the variable $x=\lambda - \mu$ and killed at $x=L$:

 $$ \mathcal{L} f(x) = -\beta x \partial_x f(x) + (x+\mu) \left[f(x+\alpha) \Theta(L-\alpha -x) - f(x)\right]   $$

 For $f(x,t) = e^{-rt} \psi(x)$


  $$ -r \psi(x) = -\beta x \partial_x \psi(x) + (x+\mu) \left[\psi (x+\alpha) \Theta(L-\alpha -x) - \psi(x)\right]   $$

Defining $\psi_j(x)$ as $\psi(x)$ in the interval $(L-(j+1)\alpha, L - j \alpha)$, we can write $x=z+L-(j+1)\alpha$ then:

  $$ -r \psi(z+L-(j+1)\alpha) = -\beta (z+L-(j+1)\alpha) \partial_z \psi(z+L-(j+1)\alpha) + (z+L-(j+1)\alpha+\mu) \left[\psi (z+L-(j+1)\alpha+\alpha) \Theta(L-\alpha -(z+L-(j+1)\alpha)) - \psi(z+L-(j+1)\alpha)\right]   $$


  $$ - r \psi_j(z) = - \beta (z + L - (j+1)\alpha) \partial_z \psi_j(z) + (z+L-(j+1)\alpha + \mu) (\psi_{j-1}(z) - \psi_j(z))

Let the forward SDE be

$$
d\boldsymbol{x}_t=-\kappa_t\boldsymbol{x}_t\,dt+g_t\,d\boldsymbol{W}_t,\qquad \boldsymbol{x}_0\sim q.
$$

Define

$$
\alpha_t=\exp\left(-\int_0^t\kappa_s\,ds\right),\qquad \dot{\alpha}_t=-\kappa_t\alpha_t.
$$

For VP we have $\kappa_t = \frac{1}{2}\beta_0$ and $\alpha_t = e^{-\frac{1}{2} \beta_0 t}$ and $g_t=\beta_0$. 
The solution is

$$
\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\alpha_t\int_0^t\frac{g_s}{\alpha_s}\,d\boldsymbol{W}_s.
$$

Therefore,

$$
p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)=\mathcal N\left(\alpha_t\boldsymbol{x}_0,\bar{\sigma}_t^2I\right),
$$

where

$$
\bar{\sigma}_t^2=\alpha_t^2\int_0^t\frac{g_s^2}{\alpha_s^2}\,ds,\qquad \frac{d\bar{\sigma}_t^2}{dt}=-2\kappa_t\bar{\sigma}_t^2+g_t^2,\qquad \bar{\sigma}_0^2=0.
$$

The marginal density is

$$
p_t(\boldsymbol{x})=\int\frac{d^dx_0}{(2\pi\bar{\sigma}_t^2)^{d/2}}\,q(\boldsymbol{x}_0)\exp\left[-\frac{\|\boldsymbol{x}-\alpha_t\boldsymbol{x}_0\|^2}{2\bar{\sigma}_t^2}\right].
$$

The score is

$$
\boldsymbol{s}_t(\boldsymbol{x})=\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}).
$$

Differentiating the Gaussian kernel gives

$$
\boldsymbol{s}_t(\boldsymbol{x})=-\frac{\boldsymbol{x}}{\bar{\sigma}_t^2}+\frac{\alpha_t}{\bar{\sigma}_t^2}\frac{\int d^dx_0\,\boldsymbol{x}_0q(\boldsymbol{x}_0)\exp\left[-\frac{\|\boldsymbol{x}-\alpha_t\boldsymbol{x}_0\|^2}{2\bar{\sigma}_t^2}\right]}{\int d^dx_0\,q(\boldsymbol{x}_0)\exp\left[-\frac{\|\boldsymbol{x}-\alpha_t\boldsymbol{x}_0\|^2}{2\bar{\sigma}_t^2}\right]}.
$$

Equivalently,

$$
\boldsymbol{s}_t(\boldsymbol{x})=-\frac{\boldsymbol{x}-\alpha_t\mathbb E[\boldsymbol{x}_0\mid\boldsymbol{x}_t=\boldsymbol{x}]}{\bar{\sigma}_t^2}.
$$

The probability-flow ODE is

$$
\frac{d\boldsymbol{x}_t}{dt}=-\kappa_t\boldsymbol{x}_t-\frac{g_t^2}{2}\boldsymbol{s}_t(\boldsymbol{x}_t).
$$

For sampling, this ODE is integrated backward from $t=T$ to $t=0$, starting from

$$
\boldsymbol{x}_T\sim p_T.
$$

Substituting the exact score gives

$$
\frac{d\boldsymbol{x}_t}{dt}=-\kappa_t\boldsymbol{x}_t+\frac{g_t^2}{2\bar{\sigma}_t^2}\left(\boldsymbol{x}_t-\alpha_t\mathbb E[\boldsymbol{x}_0\mid\boldsymbol{x}_t]\right).
$$

For the empirical distribution

$$
q(\boldsymbol{x}_0)=\frac{1}{K}\sum_{\mu=1}^K\delta(\boldsymbol{x}_0-\boldsymbol{\xi}_\mu),
$$

the empirical score is

$$
\boldsymbol{s}_t(\boldsymbol{x})=-\frac{\boldsymbol{x}}{\bar{\sigma}_t^2}+\frac{\alpha_t}{\bar{\sigma}_t^2}\frac{\sum_{\mu=1}^K\boldsymbol{\xi}_\mu\exp\left[-\frac{\|\boldsymbol{x}-\alpha_t\boldsymbol{\xi}_\mu\|^2}{2\bar{\sigma}_t^2}\right]}{\sum_{\mu=1}^K\exp\left[-\frac{\|\boldsymbol{x}-\alpha_t\boldsymbol{\xi}_\mu\|^2}{2\bar{\sigma}_t^2}\right]}.
$$

Now apply the transformation

$$
\boldsymbol{y}_t=\frac{\boldsymbol{x}_t}{\alpha_t}.
$$

Differentiating gives

$$
\frac{d\boldsymbol{y}_t}{dt}=\frac{1}{\alpha_t}\frac{d\boldsymbol{x}_t}{dt}-\frac{\dot{\alpha}_t}{\alpha_t^2}\boldsymbol{x}_t.
$$

Using $\dot{\alpha}_t=-\kappa_t\alpha_t$, the linear drift cancels:

$$
\frac{d\boldsymbol{y}_t}{dt}=-\frac{g_t^2}{2\alpha_t}\boldsymbol{s}_t(\alpha_t\boldsymbol{y}_t)=-\frac{g_t^2}{2\alpha_t^2}\left[\alpha_t\boldsymbol{s}_t(\alpha_t\boldsymbol{y}_t)\right].
$$

For the empirical score,

$$
\alpha_t\boldsymbol{s}_t(\alpha_t\boldsymbol{y})=\frac{\alpha_t^2}{\bar{\sigma}_t^2}\left[-\boldsymbol{y}+\frac{\sum_{\mu=1}^K\boldsymbol{\xi}_\mu\exp\left[-\frac{\alpha_t^2}{2\bar{\sigma}_t^2}\|\boldsymbol{y}-\boldsymbol{\xi}_\mu\|^2\right]}{\sum_{\mu=1}^K\exp\left[-\frac{\alpha_t^2}{2\bar{\sigma}_t^2}\|\boldsymbol{y}-\boldsymbol{\xi}_\mu\|^2\right]}\right].
$$

Hence,

$$
\frac{d\boldsymbol{y}_t}{dt}=-\frac{g_t^2}{2\bar{\sigma}_t^2}\left[-\boldsymbol{y}_t+\frac{\sum_{\mu=1}^K\boldsymbol{\xi}_\mu\exp\left[-\frac{\alpha_t^2}{2\bar{\sigma}_t^2}\|\boldsymbol{y}_t-\boldsymbol{\xi}_\mu\|^2\right]}{\sum_{\mu=1}^K\exp\left[-\frac{\alpha_t^2}{2\bar{\sigma}_t^2}\|\boldsymbol{y}_t-\boldsymbol{\xi}_\mu\|^2\right]}\right].
$$

Define the transformed variance

$$
\rho_t^2=\frac{\bar{\sigma}_t^2}{\alpha_t^2}=\int_0^t\frac{g_s^2}{\alpha_s^2}\,ds.
$$

Define the corresponding precision

$$
\lambda_t=\frac{1}{\rho_t^2}=\frac{\alpha_t^2}{\bar{\sigma}_t^2}=\left(\int_0^t\frac{g_s^2}{\alpha_s^2}\,ds\right)^{-1}.
$$

For VP $\lambda_t^{-1} = \int_0^t \frac{g_s^2}{\alpha_s^2} ds = \int_0^t \beta_0^2 e^{\beta_0 s} = \beta_0 (e^{\beta_0 t}-1)  $ also $\lambda_t =\frac{1}{\beta_0(e^{\beta_0 t}-1)}$. 

Introduce

$$
\boldsymbol{m}_{\lambda}(\boldsymbol{y})=\frac{\sum_{\mu=1}^K\boldsymbol{\xi}_\mu\exp\left[-\frac{\lambda}{2}\|\boldsymbol{y}-\boldsymbol{\xi}_\mu\|^2\right]}{\sum_{\mu=1}^K\exp\left[-\frac{\lambda}{2}\|\boldsymbol{y}-\boldsymbol{\xi}_\mu\|^2\right]}.
$$

Then

$$
\frac{d\boldsymbol{y}_t}{dt}=-\frac{g_t^2}{2\bar{\sigma}_t^2}\left[-\boldsymbol{y}_t+\boldsymbol{m}_{\lambda_t}(\boldsymbol{y}_t)\right].
$$

Since

$$
\dot{\lambda}_t=-\frac{g_t^2}{\alpha_t^2}\lambda_t^2,\qquad \frac{g_t^2}{\bar{\sigma}_t^2}=-\frac{\dot{\lambda}_t}{\lambda_t},
$$

we obtain

$$
\boxed{\frac{d\boldsymbol{y}_t}{dt}=\frac{\dot{\lambda}_t}{2\lambda_t}\left[-\boldsymbol{y}_t+\boldsymbol{m}_{\lambda_t}(\boldsymbol{y}_t)\right]}.
$$

Equivalently,

$$
\boxed{\frac{d\boldsymbol{y}_t}{dt}=-\frac{\dot{\lambda}_t}{2\lambda_t}\left[\boldsymbol{y}_t-\boldsymbol{m}_{\lambda_t}(\boldsymbol{y}_t)\right]}.
$$

$$
\boxed{\frac{d\boldsymbol{y}_t}{dt}=\left(\frac{d}{dt} \ln \sqrt{\lambda_t} \right)\left[\boldsymbol{m}_{\lambda_t}(\boldsymbol{y}_t)-\boldsymbol{y}_t\right]}
$$

$$
\boxed{\frac{d\boldsymbol{y}}{d\ln \sqrt{\lambda}}=\left[\boldsymbol{m}_{\lambda}(\boldsymbol{y})-\boldsymbol{y}\right]}
$$

For $\lambda \to 0$ formally $y$ has no finite moments but we can start from some $\lambda_{\rm min}$ with $ \boldsymbol{y}_{\lambda_{\rm min}} \sim \mathcal{N} \left(\bar{\boldsymbol{\xi}}, \frac{1}{\lambda_{\rm min}}  \boldsymbol{I} \right)  $. 

In summary we mapped the DDPM probability flow ODE to a MHN-like gradient descent. Indeed, for each "time" $\lambda$ the associate energy function to $\boldsymbol{y}$ is:

$$E(\boldsymbol{y},\lambda) = - \frac{1}{\lambda} \log \left(\sum_{\mu=1}^K e^{-\frac{\lambda}{2} ||\boldsymbol{y}-\boldsymbol{\xi}_\mu||^2}\right)$$

As such, at fixed $\lambda$, the attractors of the denoising dynamics coincide with the minima of the associated MHN-like energy. Those points $\boldsymbol{y}^*$ can be also studied via the Dual formulation, with a potential:
$$\Phi(\boldsymbol{w}) = \frac{1}{2} \boldsymbol{w}^\intercal \boldsymbol{G} \boldsymbol{w} - \frac{1}{2} \sum_{\mu=1}^K w_\mu \left(\frac{1}{K} \sum_\nu G_{\mu \nu}\right) - \frac{1}{\lambda} \sum_{\mu=1}^K  w_\mu \log w_\mu$$

The second bias-like term is what is absent in classic MHN energies, since it is the one that remove the implicit bias in MHN of large-normed patterns.
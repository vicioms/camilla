$$ E(\boldsymbol{x}) = -\frac{1}{\beta} \log \left(\sum_{\mu=1}^K e^{-\frac{\beta}{2}||\boldsymbol{x}-\boldsymbol{\xi}_\mu||^2}\right) $$
If we interpret $E(\boldsymbol{x})$ as a free energy from a marginalized bipartite system, we have:
$$ Z(\boldsymbol{x}) = \sum_{\mu=1}^K e^{-\frac{\beta}{2}||\boldsymbol{x}-\boldsymbol{\xi}_\mu||^2}$$
Now the question is: which is the microscopic energy which gives rise to this marginalized partition function? We can formally think that we have a $K$-Potts variable
$$Z=\int d^N x \sum_{\sigma=1}^K e^{\beta \boldsymbol{x}^\intercal \boldsymbol{\xi}_\sigma - \frac{\beta}{2} ||\boldsymbol{x}||^2 - \frac{\beta }{2} |\boldsymbol{\xi}_\sigma||^2 }$$
If we integrate out $\boldsymbol{x}$ the result becomes trivial. However, we can first replicate the PF:

$$Z^n=\int d^N x_a \sum_{\sigma_a=1}^K e^{\beta \sum_{a=1}^n \boldsymbol{x}_a^\intercal \boldsymbol{\xi}_{\sigma_a} - \frac{\beta}{2} \sum_{a=1}^n ||\boldsymbol{x}_a||^2 - \sum_{a=1}^n \frac{\beta }{2} ||\boldsymbol{\xi}_{\sigma_a}||^2 }$$

Using indicators:

$$Z^n=\int d^N x_a \sum_{\sigma_a=1}^K e^{\beta \sum_{\mu=1}^K \sum_{a=1}^n \boldsymbol{x}_a^\intercal  \boldsymbol{\xi}_{\mu} \delta_{\mu, \sigma_a} - \frac{\beta}{2}  ||\boldsymbol{x}_a||^2 -  \frac{\beta }{2}  \sum_{\mu=1}^K ||\boldsymbol{\xi}_{\mu}||^2 \sum_{a=1}^n \delta_{\mu,\sigma_a} }$$

$$S_{\mu a} = \delta_{\mu, \sigma_a}$$

$$Z^n=\int d^N x_a \sum_{\sigma_a=1}^K e^{\beta \sum_{\mu=1}^K \sum_{a=1}^n \boldsymbol{x}_a^\intercal  \boldsymbol{\xi}_{\mu} S_{\mu a} - \frac{\beta}{2}  ||\boldsymbol{x}_a||^2 -  \frac{\beta }{2}  \sum_{\mu=1}^K  \sum_{a=1}^n ||\boldsymbol{\xi}_{\mu}||^2 S_{\mu a} }$$

The disorder average yields:
$$ -\frac{\beta}{2} \sum_a ||\boldsymbol{x}_a||^2 - \frac{N}{2} \ln \det (\delta_{\mu \nu} +  \beta \sum_a C_{\mu \nu} S_{\mu a}) + \frac{\beta^2}{2} \sum_{ab\mu \nu i} (C_{\mu\nu}^{-1} + \delta_{\mu \nu} \sum_c S_{\mu c})^{-1} S_{\mu a} S_{\nu b} x_{ai} x_{bi} $$

Using $S_{\mu a} S_{\nu a} = \sum_{a=1}^n \delta_{\mu \sigma_a} \delta_{\nu \sigma_a} = \delta_{\mu \nu} \sum_a \delta_{\mu \sigma_a}$:

$$ -\frac{\beta}{2} \sum_a ||\boldsymbol{x}_a||^2 - \frac{N}{2} \ln \det (\boldsymbol{I}_K +  \beta \boldsymbol{S} \boldsymbol{S}^\intercal \boldsymbol{C} ) + \frac{\beta^2}{2} \operatorname{Tr} \boldsymbol{x}^\intercal \boldsymbol{S}^\intercal (\boldsymbol{C}^{-1} + \beta \boldsymbol{S} \boldsymbol{S}^\intercal )^{-1} \boldsymbol{S} \boldsymbol{x} $$

Using overlap and integrating $\boldsymbol{x}_a$ out

$$ \frac{i N}{2} \operatorname{Tr}(\boldsymbol{\tilde{Q}} \boldsymbol{Q}) - \frac{N}{2} \ln \det (\boldsymbol{I}_K +  \beta \boldsymbol{S} \boldsymbol{S}^\intercal \boldsymbol{C} ) + \frac{\beta^2 N}{2} \operatorname{Tr} \boldsymbol{Q} \boldsymbol{S}^\intercal (\boldsymbol{C}^{-1} + \beta \boldsymbol{S} \boldsymbol{S}^\intercal )^{-1} \boldsymbol{S} + $$
$$ -\frac{N}{2} \ln \det (\beta \boldsymbol{I}_n + i \boldsymbol{\tilde{Q}}) $$

Integrating out the conjugate overlap matrix and applying Sylvester's on the determinant:

$$  - \frac{N}{2} \ln \det (\boldsymbol{I}_n +  \beta \boldsymbol{S}^\intercal \boldsymbol{C} \boldsymbol{S} ) + \frac{\beta^2 N}{2} \operatorname{Tr} \boldsymbol{Q} \boldsymbol{S}^\intercal (\boldsymbol{C}^{-1} + \beta \boldsymbol{S} \boldsymbol{S}^\intercal )^{-1} \boldsymbol{S} + $$
$$ +\frac{N}{2} \ln \det \boldsymbol{Q} - \frac{\beta N}{2} \operatorname{Tr} \boldsymbol{Q} $$

Using Woodbury on the inverse:

$$ -\frac{N}{2}\ln\det\left(
\boldsymbol I_n+\beta\boldsymbol S^\intercal\boldsymbol C\boldsymbol S
\right)
+\frac{N}{2}\ln\det\boldsymbol Q
-\frac{\beta N}{2}
\operatorname{Tr}\left[
\boldsymbol Q
\left(
\boldsymbol I_n+\beta\boldsymbol S^\intercal\boldsymbol C\boldsymbol S
\right)^{-1}
\right]. $$

REPLICA DEP SOURCES:
$$ Z(\boldsymbol{J}) =  \left(\frac{2\pi}{\beta}\right)^{N/2} \sum_{\mu=1}^K e^{\boldsymbol J^\intercal \boldsymbol \xi_\mu + \frac{1}{2\beta} ||\boldsymbol J||^2} $$

$$ \overline{Z(\boldsymbol{J}_a)^n}=  \left(\frac{2\pi}{\beta}\right)^{nN/2} \sum_{\mu_a=1}^K e^{ \frac{1}{2} \sum_{i\mu\nu a b} J_{ai} J_{bi} C_{\mu \nu} \delta_{\mu \mu_a} \delta_{\nu \mu_b} + \frac{1}{2\beta} \sum_a ||\boldsymbol J_a||^2} $$

Using $S_{\mu a}$ as before:

$$ \overline{Z(\boldsymbol{J}_a)^n}=  \left(\frac{2\pi}{\beta}\right)^{nN/2} \sum_{\mu_a=1}^K e^{ \frac{1}{2} \operatorname{Tr}( \boldsymbol C \boldsymbol{S} \boldsymbol J \boldsymbol J^\intercal \boldsymbol{S}^\intercal ) + \frac{1}{2\beta} \sum_a ||\boldsymbol J_a||^2} $$

Using a single source and occupation variables:

$$ \overline{Z(\boldsymbol{J}_a)^n}=  \left(\frac{2\pi}{\beta}\right)^{nN/2} \sum_{m_\mu=0, \sum_\mu m_\mu=n}^\infty \frac{n!}{\prod_\mu m_\mu!} e^{  \frac{||\boldsymbol J||^2}{2} \operatorname{Tr}(  \boldsymbol m^\intercal\boldsymbol C \boldsymbol m  ) + \frac{n}{2\beta} ||\boldsymbol J||^2} $$






Now using a HS

$$ \overline{Z(\boldsymbol{J}_a)^n}=  \left(\frac{2\pi}{\beta}\right)^{nN/2} \left(\frac{1}{(2\pi)^{K/2} \sqrt{\det \boldsymbol C}}\right)\int d^K \phi \sum_{m_\mu=0, \sum_\mu m_\mu=n}^\infty \frac{n!}{\prod_\mu m_\mu!} e^{ -\frac{1}{2} \boldsymbol \phi^\intercal \boldsymbol C^{-1} \boldsymbol \phi + ||\boldsymbol J|| \boldsymbol \phi^\intercal \boldsymbol m   + \frac{n}{2\beta} ||\boldsymbol J||^2} $$

and introducing a Larange multiplier:

$$ \overline{Z(\boldsymbol{J}_a)^n}=  \left(\frac{2\pi}{\beta}\right)^{nN/2} \int \frac{d\lambda}{2\pi} \left(\frac{1}{(2\pi)^{K/2} \sqrt{\det \boldsymbol C}}\right)\int d^K \phi \sum_{m_\mu=0, \sum_\mu m_\mu=n}^\infty \frac{n!}{\prod_\mu m_\mu!} e^{ -\frac{1}{2} \boldsymbol \phi^\intercal \boldsymbol C^{-1} \boldsymbol \phi + ||\boldsymbol J|| \boldsymbol \phi^\intercal \boldsymbol m   + \frac{n}{2\beta} ||\boldsymbol J||^2 + i \lambda (\sum_\mu m_\mu-n)} $$

Summing over:

$$ \overline{Z(\boldsymbol{J}_a)^n}=  n! \left(\frac{2\pi}{\beta}\right)^{nN/2} \int \frac{d\lambda}{2\pi} e^{- i \lambda n } \int d^K \phi e^{ -\frac{1}{2} \boldsymbol \phi^\intercal \boldsymbol C^{-1} \boldsymbol \phi + \sum_\mu \exp(i \lambda + ||\boldsymbol J||  \phi_\mu) -\frac{1}{2} \ln \det \boldsymbol C -\frac{K}{2} \ln \det (2\pi) } $$











Using $A_{J} = \left(\frac{2\pi}{\beta}\right)^{N/2} \exp\left(\frac{||\boldsymbol J||^2}{2\beta}\right) $
$$\mathcal{G}(t) = \left(\frac{1}{(2\pi)^{K/2} \sqrt{\det \boldsymbol C}}\right)\int d^K \phi \exp \left(-\frac{1}{2} \boldsymbol \phi^\intercal \boldsymbol C^{-1} \boldsymbol \phi  - t A_J \sum_{\mu=1}^K e^{||\boldsymbol J|| \phi_\mu }\right)  $$

Choosing $t = e^{-s N}$:

$$\left(\frac{1}{(2\pi)^{K/2} \sqrt{\det \boldsymbol C}}\right)\int d^K \phi \exp \left(-\frac{1}{2} \boldsymbol \phi^\intercal \boldsymbol C^{-1} \boldsymbol \phi  - A_J \sum_{\mu=1}^K e^{||\boldsymbol J|| \phi_\mu - s N}\right)  $$

We now rescale $\phi_\mu = \sqrt{N} \psi_\mu$, define $q = ||\boldsymbol J||/\sqrt{N}$  and $a(q) = \frac{\log A_J}{N} = \frac{1}{2} \log \frac{2\pi}{\beta} + \frac{q^2}{2\beta} $ 

$$\left(\frac{N^{K/2}}{(2\pi)^{K/2} \sqrt{\det \boldsymbol C}}\right)\int d^K \psi \exp \left(-\frac{N}{2} \boldsymbol \psi^\intercal \boldsymbol C^{-1} \boldsymbol \psi  - \sum_{\mu=1}^K e^{N q \psi_\mu  - s N + N a(q) }\right)  $$

The first case is $\boldsymbol C = \boldsymbol I$:

$$\left(\left(\frac{N^{1/2}}{(2\pi)^{1/2}}\right)\int d \psi \exp \left(-\frac{N}{2} \psi^2  - e^{N q \psi -  s N + N a(q) }\right) \right)^K = I_N^K  $$

Since $K \sim e^{\alpha N}$ we need to control properly the integrand inside:

$$\Delta_N =1 - I_N = \sqrt{\frac{N}{2\pi}} \int d\psi e^{-N\frac{\psi^2}{2}} \left[1-e^{-e^{N(q \psi - s + a(q))}}\right] $$


At exponential accuracy:

$$1-e^{-e^{N(q \psi - s + a(q))}} \approx e^{-N (s-a(q)-q\psi )_+}$$

Hence:

$$ \Delta_N \approx e^{-N R(q,s)} $$

with $u=s-a(q)$ and:

$$ R(q,s) = \min_{\psi} \left[\frac{\psi^2}{2} + (u - q \psi)_+\right] $$

For $u>0$, we either have $u > q \psi$ and the minimizer is $\psi = q$ provided $u \geq q^2$ or $\psi=u/q$ for $0 \leq u \leq q^2$ so:

$$ R(q,s) = \begin{cases}
\frac{u^2}{2q^2} & 0 \leq u \leq q^2 \\
u - \frac{q^2}{2} & u \geq q^2
\end{cases}$$

For $u < 0$ we need to  look at $I_N$ and we find that $\psi = u/q$ is the dominant point, yielding:

$$ I_N \approx e^{-N \frac{u^2}{2q^2}} $$

$$ I_N(u)\approx
\begin{cases}
\displaystyle
\exp\left[-N\frac{u^2}{2q^2}\right],
& u<0,
\\[3mm]
\displaystyle
\frac12,
& u=0,
\\[3mm]
\displaystyle
1-\exp\left[-N\frac{u^2}{2q^2}\right],
& 0<u\leq q^2,
\\[3mm]
\displaystyle
1-\exp\left[-N\left(u-\frac{q^2}{2}\right)\right],
& u\geq q^2.
\end{cases} $$

Now $u = s - a(q) = s - \frac{1}{2}\log \frac{2\pi}{\beta} - \frac{q^2}{2\beta} $





In this interpretation we can actually write:

$$P(\boldsymbol{x}|\beta) = \left(\frac{\beta}{2\pi}\right)^{N/2} \sum_{\mu=1}^K e^{-\frac{\beta}{2} ||\boldsymbol{x}-\boldsymbol{\xi_\mu}||^2}$$

The overlap at two different $\beta$ is:

$$ O_{\beta,\beta'} = \left(\frac{\beta}{2\pi}\right)^{N/2} \left(\frac{\beta'}{2\pi}\right)^{N/2} \int d^N \boldsymbol{x} \sum_{\mu_1,\mu_2=1}^K e^{-\frac{\beta}{2} ||\boldsymbol x - \boldsymbol \xi_{\mu_1}||^2-\frac{\beta'}{2} ||\boldsymbol x - \boldsymbol \xi_{\mu_2}||^2} $$

$$ O_{\beta,\beta'}
=
\left(
\frac{\beta\beta'}{2\pi(\beta+\beta')}
\right)^{N/2}
\sum_{\mu_1,\mu_2=1}^{K}
\exp\left[
-\frac{\beta\beta'}{2(\beta+\beta')}
\left\|\boldsymbol\xi_{\mu_1}-\boldsymbol\xi_{\mu_2}\right\|^2
\right]. $$

The $D = - \log O_{\beta,\beta'}$:

$$ D  = - \frac{N}{2} \log \frac{\beta\beta'}{\beta+\beta'} - \frac{N}{2} \log 2\pi - \log \left(\sum_{\mu_1,\mu_2=1}^{K}
\exp\left[
-\frac{\beta\beta'}{2(\beta+\beta')}
\left\|\boldsymbol\xi_{\mu_1}-\boldsymbol\xi_{\mu_2}\right\|^2
\right]\right) $$


$$
\left.
\frac{\partial^2 D}{\partial\beta\,\partial\beta'}
\right|_{\beta'=\beta}
=
-\frac{N}{8\beta^2}
+
\frac{1}{8\beta}
\frac{
\displaystyle\sum_{\mu_1,\mu_2=1}^{K}
r_{\mu_1\mu_2}^{2}
e^{-\frac{\beta}{4}r_{\mu_1\mu_2}^{2}}
}{
\displaystyle\sum_{\mu_1,\mu_2=1}^{K}
e^{-\frac{\beta}{4}r_{\mu_1\mu_2}^{2}}
}
-
\frac{1}{64}
\left[
\frac{
\displaystyle\sum_{\mu_1,\mu_2=1}^{K}
r_{\mu_1\mu_2}^{4}
e^{-\frac{\beta}{4}r_{\mu_1\mu_2}^{2}}
}{
\displaystyle\sum_{\mu_1,\mu_2=1}^{K}
e^{-\frac{\beta}{4}r_{\mu_1\mu_2}^{2}}
}
-
\left(
\frac{
\displaystyle\sum_{\mu_1,\mu_2=1}^{K}
r_{\mu_1\mu_2}^{2}
e^{-\frac{\beta}{4}r_{\mu_1\mu_2}^{2}}
}{
\displaystyle\sum_{\mu_1,\mu_2=1}^{K}
e^{-\frac{\beta}{4}r_{\mu_1\mu_2}^{2}}
}
\right)^2
\right],
\qquad
r_{\mu_1\mu_2}^{2}
=
\left\|\boldsymbol\xi_{\mu_1}-\boldsymbol\xi_{\mu_2}\right\|^2. $$

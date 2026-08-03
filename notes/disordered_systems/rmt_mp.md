$$ S = - \frac{iz}{2} \sum_{\mu a} \phi_{\mu a}^2 + \frac{i}{2} \sum_{a \mu \nu} \boldsymbol{\xi}_\mu \cdot \boldsymbol{\xi}_\nu \phi_{\mu a} \phi_{\nu a}  $$

Averaging over the disorder:
$$ S = - \frac{iz}{2} \sum_{\mu a} \phi_{\mu a}^2 -\frac{N}{2} \ln \det \left(\boldsymbol{I}_K - \frac{i}{N}  (\boldsymbol{C}^{1/2} \boldsymbol{\phi})  (\boldsymbol{C}^{1/2} \boldsymbol{\phi})^\intercal \right)$$

Using Sylvester:
$$ S = - \frac{iz}{2} \sum_{\mu a} \phi_{\mu a}^2 -\frac{N}{2} \ln \det \left(\boldsymbol{I}_n - \frac{i}{N} \boldsymbol{\phi}^\intercal \boldsymbol{C} \boldsymbol{\phi} \right)  $$

Using an overlap:

$$ S = - \frac{iz}{2} \sum_{\mu a} \phi_{\mu a}^2 -\frac{N}{2} \ln \det \left(\boldsymbol{I}_n - i\frac{K}{N} \boldsymbol{Q} \right) - i \frac{K}{2} \operatorname{Tr}\left(\boldsymbol{Q}\boldsymbol{\tilde{Q}}\right) + $$
$$ + \frac{i}{2} \sum_{ab\mu} \tilde{Q}_{ab} (\boldsymbol{C}^{1/2} \boldsymbol{\phi})_{\mu a} (\boldsymbol{C}^{1/2} \boldsymbol{\phi})_{\mu b} $$

Now let us assume that $\boldsymbol{C}$ is diagonalizable with eigenvalues $c_\mu$ (even repeated). Thus we can change variable:

$$ S = - \frac{iz}{2} \sum_{\mu a} \psi_{\mu a}^2 -\frac{N}{2} \ln \det \left(\boldsymbol{I}_n - i\frac{K}{N} \boldsymbol{Q} \right) - i \frac{K}{2} \operatorname{Tr}\left(\boldsymbol{Q}\boldsymbol{\tilde{Q}}\right) + $$
$$ + \frac{i}{2} \sum_{ab\mu} c_\mu  \tilde{Q}_{ab} \psi_{\mu a} \psi_{\mu b}  $$

Now the integral in $\psi_{\mu a}$ is trivial:

$$ S =  -\frac{N}{2} \ln \det \left(\boldsymbol{I}_n - i\frac{K}{N} \boldsymbol{Q} \right) - i \frac{K}{2} \operatorname{Tr}\left(\boldsymbol{Q}\boldsymbol{\tilde{Q}}\right) - \frac{1}{2} \sum_{\mu=1}^K \ln \det \left(z \boldsymbol{I} - c_\mu \boldsymbol{\tilde{Q}} \right) $$

Using $\alpha = K/N$:

$$ S =  -\frac{K}{2\alpha} \ln \det \left(\boldsymbol{I}_n - i \alpha \boldsymbol{Q} \right) - i \frac{K}{2} \operatorname{Tr}\left(\boldsymbol{Q}\boldsymbol{\tilde{Q}}\right) - \frac{1}{2} \sum_{\mu=1}^K \ln \det \left(z \boldsymbol{I}_n - c_\mu \boldsymbol{\tilde{Q}} \right) $$

Taking the saddle point:

$$ \left(\boldsymbol{I}_n - i \alpha \boldsymbol{Q} \right)^{-1} - \boldsymbol{\tilde{Q}} = 0$$

$$ i \boldsymbol{Q} = \frac{1}{K}  \sum_{\mu=1}^K c_\mu (z \boldsymbol{I}_n - c_\mu \tilde{Q})^{-1} $$

Eliminating $\boldsymbol{Q}$:

$$  i \boldsymbol{Q}  = \frac{ \boldsymbol{I}_n-\boldsymbol{\tilde{Q}}^{-1}}{\alpha}$$

$$ \boldsymbol{\tilde{Q}} - \boldsymbol{I}_n = \frac{\alpha}{K}  \sum_{\mu=1}^K c_\mu \boldsymbol{\tilde{Q}} \, (z \boldsymbol{I}_n - c_\mu \tilde{Q})^{-1} $$

$$ \frac{S}{K} =  \frac{1}{2\alpha} \ln \det \boldsymbol{\tilde{Q}} - \frac{1}{2\alpha} \operatorname{Tr}\boldsymbol{\tilde{Q}} - \frac{1}{2K} \sum_{\mu=1}^K \ln \det \left(z \boldsymbol{I}_n - c_\mu \boldsymbol{\tilde{Q}} \right) $$

Using diagonal RS:

$$ 1 - \tilde{q}(z) =  \alpha - \frac{\alpha z}{K} \sum_{\mu=1}^K \frac{1}{z-c_\mu \tilde{q}(z)} $$

$$ \frac{\tilde{q}(z)- 1 + \alpha }{\alpha z}=  \frac{1}{K} \sum_{\mu=1}^K \frac{1}{z-c_\mu \tilde{q}(z)} $$

Let us start with equi-correlated matrices. Then $c_\mu = 1 - \rho$ with multiplicity $K-1$ and $1+(K-1)\rho$ with $1$. Thus:

$$ \frac{\tilde{q}(z)- 1 + \alpha }{\alpha z}=  \frac{K-1}{K} \frac{1}{z-(1-\rho)\tilde{q}(z)} + \frac{1}{K} \frac{1}{z-(1+(K-1)\rho) \tilde{q}(z)}  $$


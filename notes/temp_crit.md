Consider

$$\tau_i(t)=\max_j\left[\sum_{t'=0}^{t-1}A_{ij}(t,t')\tau_j(t')-B_{ij}(t)\right]_++\varepsilon_i(t).$$

Consider "Markovian" delays on an inverted binary tree, with many downstream firms producing for fewer and fewer firms:

$$\tau_{\ell,i}(t)=\max_j\left[a_{j\to i}^{\ell+1,\ell}(t-1)\tau_{\ell+1,j}(t-1)-b_{j\to i}^{\ell+1,\ell}(t)\right]_++\varepsilon_{\ell,i}(t).$$

For such a network, nodes at some level $\ell$ are affected by nodes at level $\ell+1$. Consider the non-self conductivities all equal to $1$, constant thresholds $b$, and introduce a self-conductivity $\alpha$. Calling $p_{\ell,t}(\tau)$ density of $\tau_{\ell,i}(t)$ for a generic node $i$ at level $\ell$ we introduce as $Q_{\ell, t}(\tau)$ the CDF of the maximum of $K$ delays at level $\ell$. We have:

$$ Q_{\ell, t}(\tau) = \Psi_{\ell,t}^K(\tau) $$

So:

$$ \tau_{\ell,t} = \varepsilon_{\ell, t} + \max((\alpha \tau_{\ell,t-1}-b)_+,(z_{\ell+1,t-1}-b)_+ ) $$

having called $z$ the maximum described above. The CDF of the joint maximum is:

$$ Q_{\ell+1,t-1}(M+b) F_{\ell,t-1}\left(\frac{M+b}{a}\right) $$

$$F_{\ell,t}(\tau) = \int_0^\tau d\varepsilon \, p(\varepsilon) Q_{\ell+1,t-1}(\tau - \varepsilon + b) F_{\ell,t-1}\left(\frac{\tau - \varepsilon +b}{a}\right) $$

At stationarity:

$$F_{\ell}(\tau) = \int_0^\tau d\varepsilon \, p(\varepsilon) F_{\ell+1}^K(\tau - \varepsilon + b) F_{\ell}\left(\frac{\tau - \varepsilon +b}{a}\right) $$

If $a =0 $:

$$F_{\ell}(\tau) = \int_0^\tau d\varepsilon \, p(\varepsilon) F_{\ell+1}^K(\tau - \varepsilon + b)  $$

$$F_{\ell}(\tau) = \int_0^\tau dz \, p(\tau-z) F_{\ell+1}^K(z + b)  $$

$$ \partial_\tau F_{\ell}(\tau) = p(0) F_{\ell+1}^K(\tau+b) + \int_0^\tau dz p'(\tau-z) F_{\ell+1}^K(z + b)  $$

For exponential $\varepsilon$:

$$ \partial_\tau F_{\ell}(\tau) = \nu F_{\ell+1}^K(\tau+b) -\nu  F_{\ell}(\tau)  $$

If indeed we have a typical delay for $\ell \to \infty$, we can inspect the tail of $F_{\ell}(\tau)$ as $F_{\ell}(\tau) \sim 1 - c_{\ell} e^{-\lambda_\ell \tau} $

$$ \lambda_\ell c_{\ell} e^{-\lambda_{\ell} \tau} + \nu -\nu c_\ell e^{-\lambda_{\ell}} = \nu (1-c_{\ell} e^{-\lambda_{\ell+1} \tau})^K \approx \nu(1-K c_{\ell} e^{-\lambda_{\ell+1} \tau - \lambda_{\ell+1} b}) $$

yielding:

$$ 1 - \frac{\lambda}{\nu} = K  e^{-\lambda b} $$




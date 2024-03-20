<h1>Short Notes: Implemented RL Algorithms</h1>

**Advantage Actor Critic (A2C)**

```math
L^{\pi}_{A2C}(\theta) = \hat{\mathds{E}}_{t}\Bigg[ log \pi_{\theta}(a_{t}|s_{t}) \hat{A}_{t} + \beta_{s}H \bigg(\pi(\cdot|s_{t}) \bigg) \Bigg].
```

```math
L^{v}(\phi) = \hat{\mathds{E}}_{t}\left[\frac{1}{2} \Bigg(v_{\phi}(s_{t}) - \hat{v}_{t}^{target} \Bigg)^{2}\right].
```


**Advantage Actor Critic (PPO)**

```math
L^{\pi}_{PPO}(\theta) = \hat{\mathds{E}}_{t}\Bigg[ \Bigg. min \Bigg(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}\hat{A}_{t}, clip \bigg(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}, 1-\epsilon, 1+\epsilon \bigg)\hat{A}_{t} \Bigg) + \beta_{s}H \bigg(\pi(\cdot|s_{t}) \bigg)\Bigg. \Bigg].
```

```math
L^{v}(\phi) = \hat{\mathds{E}}_{t}\left[\frac{1}{2} \Bigg(v_{\phi}(s_{t}) - \hat{v}_{t}^{target} \Bigg)^{2}\right].
```

**Advantage Actor Critic (G2P2C)**

**Advantage Actor Critic (SAC)**

```math
L^{Q}_{SAC}(\phi_{j}) = \hat{\mathds{E}}_{t}\left[\frac{1}{2} \Bigg(Q_{\phi_{j}}(s_{t}) - \hat{Q}_{t}^{target} \Bigg)^{2}\right] \text{for } j \in \{1,2\}.
```

```math
\bar{\phi_{j}} = \tau \phi_{j} + (1-\tau) \bar{\phi_{j}} \text{ for } j \in \{1,2\}.
```

```math
L^{\pi}_{SAC}(\theta) = \hat{\mathds{E}}_{t}\Bigg[ \alpha log(\pi_{\theta}(a_{t}|s_{t})) - Q_{\phi}(s_{t}, a_{t}) \Bigg].
```

```math
L^{\alpha}_{SAC}(\alpha) = \hat{\mathds{E}}_{t}\Bigg[ -\alpha log \pi_{t}(a_{t}|s_{t}) - \alpha \bar{H} \Bigg].
```




<h1>Short Notes: Implemented RL Algorithms</h1>

**Advantage Actor Critic (A2C)**

```math
L^{Q}_{SAC}(\phi_{j}) = \hat{\mathds{E}}_{t}\left[\frac{1}{2} \Bigg(Q_{\phi_{j}}(s_{t}) - \hat{Q}_{t}^{target} \Bigg)^{2}\right] \text{for } j \in \{1,2\}.
```

**Advantage Actor Critic (PPO)**

**Advantage Actor Critic (G2P2C)**

**Advantage Actor Critic (SAC)**

```math
L^{Q}_{SAC}(\phi_{j}) = \hat{\mathds{E}}_{t}\left[\frac{1}{2} \Bigg(Q_{\phi_{j}}(s_{t}) - \hat{Q}_{t}^{target} \Bigg)^{2}\right] \text{for } j \in \{1,2\}.

\bar{\phi_{j}} = \tau \phi_{j} + (1-\tau) \bar{\phi_{j}} \text{ for } j \in \{1,2\}.
```




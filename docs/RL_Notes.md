<h1>Short Notes: Implemented RL Algorithms</h1>

**Advantage Actor Critic (A2C)**<br>

![A2C_Architecure](../img/base_architecture.png)

```math
L^{\pi}_{A2C}(\theta) = \hat{E}_{t}\Bigg[ log \pi_{\theta}(a_{t}|s_{t}) \hat{A}_{t} + \beta_{s}H \bigg(\pi(\cdot|s_{t}) \bigg) \Bigg].
```

```math
L^{v}(\phi) = \hat{E}_{t}\left[\frac{1}{2} \Bigg(v_{\phi}(s_{t}) - \hat{v}_{t}^{target} \Bigg)^{2}\right].
```


**Advantage Actor Critic (PPO)**<br>

![PPO_Architecure](../img/base_architecture.png)

```math
L^{\pi}_{PPO}(\theta) = \hat{E}_{t}\Bigg[ \Bigg. min \Bigg(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}\hat{A}_{t}, clip \bigg(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}, 1-\epsilon, 1+\epsilon \bigg)\hat{A}_{t} \Bigg) + \beta_{s}H \bigg(\pi(\cdot|s_{t}) \bigg)\Bigg. \Bigg].
```

```math
L^{v}(\phi) = \hat{E}_{t}\left[\frac{1}{2} \Bigg(v_{\phi}(s_{t}) - \hat{v}_{t}^{target} \Bigg)^{2}\right].
```

**Advantage Actor Critic (G2P2C)**<br>

![G2P2C_Architecure](../img/G2P2C_architecure.png)

Consists of 3 optimisation phases, the first phase is similar to **PPO**.<br>
Model Learning Phase <br>
```math
L^{M^{\Pi}}(\theta) = \hat{E}_{t}\Bigg[ \Bigg. -log \bigg(M^{\Pi}_{\theta}(g_{t+1}|s_{t}, a_{t}) \bigg) + \beta_{1}d_{KL} \bigg[\pi_{\theta_{ppo}}(\cdot|s_{t}), \pi_{\theta}(\cdot|s_{t}) \bigg]\Bigg. \Bigg].
```
```math
L^{M^{V}}(\phi) = \hat{E}_{t}\Bigg[ \Bigg. -log \bigg(M^{V}_{\phi}(g_{t+1}|s_{t}, a_{t}) \bigg) + \beta_{2}\frac{1}{2} \bigg(v_{\phi_{ppo}}(s_{t}) - \hat{v}_{\phi}(s_{t}) \bigg)^{2}\Bigg. \Bigg].
```
Planning Phase <br>
```math
\tau^{*} = \arg \max_{\tau}\Bigg[ \Bigg. (\sum_{q=1}^{n_{plan}}R_{q}) + v(s_{n_{plan}})\Bigg].
```
```math
L^{plan}(\theta) = \hat{E}_{t}\Bigg[ \Bigg. -log \bigg(\pi_{\theta}(a^{*}_{t}|s_{t}) \bigg)\Bigg].
```


**Advantage Actor Critic (SAC)**<br>

![SAC_Architecure](../img/SAC_architecture.png)

```math
L^{Q}_{SAC}(\phi_{j}) = \hat{E}_{t}\left[\frac{1}{2} \Bigg(Q_{\phi_{j}}(s_{t}) - \hat{Q}_{t}^{target} \Bigg)^{2}\right] \text{for } j \in \{1,2\}.
```

```math
\bar{\phi_{j}} = \tau \phi_{j} + (1-\tau) \bar{\phi_{j}} \text{ for } j \in \{1,2\}.
```

```math
L^{\pi}_{SAC}(\theta) = \hat{E}_{t}\Bigg[ \alpha log(\pi_{\theta}(a_{t}|s_{t})) - Q_{\phi}(s_{t}, a_{t}) \Bigg].
```

```math
L^{\alpha}_{SAC}(\alpha) = \hat{E}_{t}\Bigg[ -\alpha log \pi_{t}(a_{t}|s_{t}) - \alpha \bar{H} \Bigg].
```




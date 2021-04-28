# Zone temperature predictor
The zone air temperature dynamics is identified using a nonlinear autoregressive exogenuous (NARX) model as shown below. 
$\hat T$ and $T$ are the predicted and measured temperature respectively.
$L$ is the lagged time steps, and here is used to capture the influence of past temperature values on the future temperature prediction.
$\dot m$ is the zonal air mass flowrate.
$\dot q$ represents error term.
$\alpha$, $\beta$ and $\gamma$ are identified coefficients from given measurement data.
Subscripts $z$, $oa$ and $j$ are the zone air, outdoor air and step index respectively.
Superscripts $t+1$ and $t-j$ represent one step in the future and $j$ steps in the past.

$$\hat T_z^{t+1} = \sum_{j=0}^{L-1} \alpha_jT_z^{t-j} + \beta\dot m_z^{t+1} + \gamma \hat T_{oa}^{t+1} + \dot q_z^{t+1} $$

$$\dot q_z^{t+1} = \sum_{j=0}^{L-1} \frac{\hat T_z^{t-j}-T_z^{t-j}}{L}$$
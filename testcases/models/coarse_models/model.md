# Zone temperature predictor


$$T_z^{t+1} = \sum_{j=0}^{l-1} \alpha_jT_z^{t-j} + \beta\dot m_z^{t+1} + \gamma T_{oa}^{t+1} + \dot q_z^{t+1} $$

$$\dot q_z^{t+1} = \sum_{j=0}^{l-1} \frac{T_z^{t-j}-\hat T_z^{t-j}}{l}$$
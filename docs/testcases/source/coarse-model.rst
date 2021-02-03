.. _chapCoarseModel:

Coarse Models
===============

Zone Dynamics
--------------

.. math:: T_{z,n}^{t} = \sum_{i=1}^{l}\alpha_i T_{z,n}^{t-i} + \beta_n\dot m_n^{t}(T_{s,n}^{t} - T_{z,n}^{t}) + \gamma T_o^t + Q_{z,n}^{t}


System Power
-------------

.. math:: P^t = \sum_{i=1}^{l}\alpha_i P^{t-i} + \sum_{j=1}^m \beta_j (\dot m_s^t)^j + \sum_{k=0}^{o} \gamma_k (T_o^t)^k
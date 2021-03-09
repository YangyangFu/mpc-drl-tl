import numpy as np

def zone_temperature(alpha, beta, gamma, l, Tz_his, mz, Ts, Toa):
    """Predicte zone temperature at next step

    :param alpha: coefficients from curve-fitting
    :type alpha: np array (l,)
    :param beta: coefficient from curve-fitting
    :type beta: scalor
    :param gamma: coefficient from curve-fitting
    :type gamma: scalor
    :param l: historical step
    :type l: scalor
    :param Tz_his: historical zone temperature array
    :type Tz_his: np array (l,)
    :param mz: zone air mass flowrate at time t
    :type mz: scalor
    :param Ts: discharge air temperaure at time t
    :type Ts: scalor
    :param Toa: outdoor air dry bulb temperature at time t
    :type Toa: scalor

    :return: predicted zone temperature at time t
    :rtype: scalor
    """
    # check dimensions
    if int(l) != len(alpha) or int(l) != len(Tz_his):
        raise ValueError("'l' is not equal to the size of historical zone temperature or the coefficients.")

    Tz = (sum(alpha*Tz_his) + beta*mz*Ts + gamma*Toa)/(1+beta*mz)
    return Tz

def total_power(alpha, beta, gamma, l, P_his, mz, Toa):
    """Predicte zone temperature at next step

    :param alpha: coefficients from curve-fitting
    :type alpha: np array (l,)
    :param beta: coefficient from curve-fitting
    :type beta: np array (2,)
    :param gamma: coefficient from curve-fitting
    :type gamma: np array (3,)
    :param l: historical step
    :type l: scalor
    :param P_his: historical zone temperature array
    :type P_his: np array (l,)
    :param mz: zone air mass flowrate at time t
    :type mz: scalor
    :param Toa: outdoor air dry bulb temperature at time t
    :type Toa: scalor

    :return: predicted system power at time t
    :rtype: scalor
    """
    # check dimensions
    if int(l) != len(alpha) or int(l) != P_his.shape[1]:
        raise ValueError("'l' is not equal to the size of historical zone temperature or the coefficients.")

    P = (np.sum(alpha*P_his,axis=1) + beta[0]*mz+beta[1]*mz**2 + gamma[0]+ gamma[1]*Toa+gamma[2]*Toa**2)
    return P

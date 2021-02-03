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


# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

"""
Some descriptions
"""

import logging
import math
import numpy as np
import pandas as pd
from gym import spaces
import os
import gym
import json
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn

class ANNSingleZoneEnv(gym.Env):
    """
    ### Description

    This environmet correspond to the reduced order model of Single Zone.  
    The zone is represented by an ANN model. The Reduced order model for FCU is polynomial model.  
    The agent (a fan coil unit system) is controlled to minimize energy cost while maintaining the zone thermal comfort. 
    For any given state the agent may choose to operate the fan at a different speed (i.e. control variable is the frequency of the fan).
    
    ### Action Space
        Type: Discret(nActions)
        |Num      |   Action              |
        |---------|-----------------------|
        |0        |   Fan off             |
        |...      |                       |
        |...      |                       |
        |nAction-1|   Fan on at full speed| 


    ### Observation Space
        Type: Box(5+3*n_next_steps+2*m_prev_steps)
        Num                                     Observation                                     Min            Max
        0                                       Time                                            0              86400
        1                                       Zone temperature                                273.15 + 12    273.15 + 35
        2                                       Outdoor temperature                             273.15 + -10     273.15 + 40
        3                                       Solar radiation                                 0              1000
        4                                       Total power                                     0              1500
        5                                       Energy price                                    0              1
        5+[1...,n_next_steps]                   Outdoor temperature prediction at next n step   273.15 + 0     273.15 + 40
        5+n_next_steps+[1,...,n_next_steps]     Solar radiation prediction at next n step       0              1000
        5+2*n_next_steps+[1,...,n_next_steps]   Energy price at next n step                     0              1
        5+3*n_next_steps+[1,...,m_prev_steps]   Zone temperature from previous m steps          273.15 + 12    273.15 + 35   
        5+3*n_next_steps+m_prev_steps+[1,...,m_prev_steps]   Total power from previous m steps  0              1500
        #total power is replaced with previous outdoor temperature
        #TODO to confirm if the previous tatal power needed and modify the obsevation space

    
    ### Rewards
         Sum of energy costs, zone temperature violations and action slew rate

    ### Starting State (Need to revise)
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End (Need to revise)
    The episode ends if any one of the following occurs:
    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)



    ### Attributes
    mass_flow_nor (float): List, norminal mass flow rate of VAV terminals.
    weather_file (str): Energyplus epw weather file name 
    n_next_steps (int): number of future prediction steps
    time_step (float): time difference between simulation steps.

    """

    def __init__(self,
                 mass_flow_nor,
                 weather_file,
                 n_next_steps,
                 simulation_start_time,
                 simulation_end_time,
                 time_step,
                 log_level,
                 alpha=0.01, # penalty coefficients for temperature violation in reward function 
                 nActions=11,
                 rf=None,
                 p_g=None,
                 # n_substeps=15,
                 m_prev_steps=0):

        logger.setLevel(log_level)

        # system parameters
        self.mass_flow_nor = mass_flow_nor 
        self.weather_file = weather_file 
        self.n_next_steps = n_next_steps 
        self.m_prev_steps = m_prev_steps

        # virtual environment simulation period
        self.simulation_start_time = simulation_start_time
        self.simulation_end_time = simulation_end_time
        self.tau = time_step # seconds between state updates
        # state bounds if any
        
        # experiment parameters
        self.alpha = alpha # Positive: penalty coefficients for temperature violation in reward function 
        self.nActions = nActions # Integer: number of actions for one control variable (level of frequency of fan coil)

        # customized reward return
        self.rf = rf # this is an external function        
        # customized hourly TOU energy price
        if not p_g:
            self.p_g = [0.02987, 0.02987, 0.02987, 0.02987, 
                        0.02987, 0.02987, 0.04667, 0.04667, 
                        0.04667, 0.04667, 0.04667, 0.04667, 
                        0.15877, 0.15877, 0.15877, 0.15877,
                        0.15877, 0.15877, 0.15877, 0.04667, 
                        0.04667, 0.04667, 0.02987, 0.02987]
        else:
            self.p_g = p_g           
        assert len(self.p_g)==24, "Daily hourly energy price should be provided!!!"

        # # number of substeps to output
        # self.n_substeps=n_substeps

        self.action_space = spaces.Discrete(self.nActions)
        self.observation_space = self._get_observation_space()
        self.state = None

        # # others
        # self.viewer = None
        # self.display = None

        # conditional
        self.history = None
        # if self.m_prev_steps > 0:
        #     self.history['TRoo'] = [273.15+25]*self.m_prev_steps
        #     self.history['PTot'] = [0.]*self.m_prev_steps

        # initialize some metadata 
        self._cost = []
        self._max_temperature_violation = []
        self._delta_action = []
        
    def _get_hour_price(self,time):
        """
        Parameters
        ----------
        time : list of int
            time in seconds.

        Returns
        -------
        price : list of double
            price at that hour.

        """
        pri = []
        for i in time:
            hour = math.ceil(i/3600) % 24
            pri.append(self.p_g[hour-1])
        return pri
        

    def step(self, action):
        """
        OpenAI Gym API. Executes one step in the environment:
        in the current state perform given action to move to the next action.
        Applies force of the defined magnitude in one of two directions, depending on the action parameter sign.

        :param action: alias of an action [0-10] to be performed. 
        :return: next (resulting) state
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        # update historical action for reward calculation
        self.action_curr = action
        # forward action
        # action = np.array(action)
        # action = [action/float(self.nActions-1)]
        action = action/float(self.nActions-1) # can only be single
        # predict future Tz and P with ROM
        Tz, P = self._rom_model(action)
        # update state
        print("State before update \n{}".format(self.state))
        self._get_observation(Tz, P)
        print("State after update \n{}".format(self.state))
        # reward policy
        rewards = self._reward_policy()
        # update action
        self.action_prev = self.action_curr
        return Tz, self.state, rewards, {}
        
        # return P
        # return super(SingleZoneEnv,self).step(action)
    
    def _rom_model(self, action):
        x = self._get_model_state(action)
        #load ann to predict Troom
        ann = Net(features=x.shape[1])
        cur_path = os.path.dirname(os.path.realpath(__file__))
        # ann = torch.load(cur_path + '/zone_ann.pt')
        state_dic = torch.load(cur_path + '/zone_ann.pt')
        ann.load_state_dict(state_dic)
        x_data = torch.tensor(x, dtype=torch.float)
        Tz = ann(x_data).detach().numpy()
        # polynomial
        # action is ufan, ufan need to convert to mass flowrate
        # m = [i * 0.55 for i in action]
        m = 0.55 * action #0.55 is the capacity, change to air flow
        with open(cur_path + '/power.json', 'r') as fcc_file:
            alpha = json.load(fcc_file)
        alpha = list(alpha.values())[0]
        Ts = 273.15 + 14
        P = alpha[0]+alpha[1]*m+alpha[2]*m**2+alpha[3]*m**3 +(1008./3)*m*(Tz[0][0]-Ts)#+ beta[0]+ beta[1]*Toa+beta[2]*Toa**2
        return Tz, P

    def _get_model_state(self, action):
        # can also use self.current_state etc.
        x = np.empty([1,4+1*self.n_next_steps+2*self.m_prev_steps], dtype = float)
        x[0,0:3] = self.state[1:4]
        m = 0.55 * action #0.55 is the capacity, change to air flow
        x[0,3] = m
        x[0,4:4+self.n_next_steps] = self.state[6:6+self.n_next_steps]
        x[0,4+self.n_next_steps:4+self.n_next_steps+2*self.m_prev_steps] = self.state[-2*self.m_prev_steps:]
        return x

    # get states of gym ENV
    def _get_observation(self,Tz, P):
        tim = self.simulation_start_time + self.tau
        # current states
        self.current_state = [tim, Tz[0][0], self.future_state[0], self.future_state[self.n_next_steps], P
                         , self.future_state[self.n_next_steps*2]]
        # get price in this time period
        tims = []
        for i in range(self.m_prev_steps):
            tims.append(tim - self.tau * (i+1))
        tims = tims[::-1]
        tims.append(tim)
        for i in range(self.n_next_steps):
            tims.append(tims[-1] + self.tau)
        ene_pri = self._get_hour_price(tims)        
        # future states
        out_tem_sol_n = self.predictor(self.n_next_steps)  # outdoor air temperature for next, n * 2, temperature first, solar in the end
        self.future_state = list(out_tem_sol_n) + list(ene_pri[-self.n_next_steps:])
        
        # history states
        self.history_state[0:self.m_prev_steps-1] = self.history_state[1:self.m_prev_steps]
        self.history_state[self.m_prev_steps-1] = self.state[1]
        self.history_state[self.m_prev_steps:self.m_prev_steps*2-1] = self.history_state[self.m_prev_steps+1:self.m_prev_steps*2]
        self.history_state[self.m_prev_steps*2-1] = self.state[2]
        
        self.state = tuple(self.current_state + self.future_state + self.history_state)
        # (tim, zon_tem_mk[-1], out_tem_mk[-1], ghi_mk[-1], power
        #               , ene_pri[self.m_prev_steps], out_tem_sol_n)
        assert len(self.state)==self.observation_space.shape[0], "The size of state is not consistent with observation space"

    def reset(self):
        """
        Inherit the existing internal reset method and customize for this environment
        """
        # if self.m_prev_steps > 0:
        #     self.history['TRoo'] = [273.15+25]*self.m_prev_steps 
        #     self.history['PTot'] = [0.]*self.m_prev_steps
        # # reset previous action to calculate rewards in terms of action changes during steps
        
        # time, zone_tem_mk (history and current zone temperature), out_wea_sol_mk (history and current outdoor temperature),
        # out_wea_sol_n (future outdoor temperature)
        tim = self.simulation_start_time
        his_data_path = os.path.dirname(os.path.realpath(__file__))
        history_data = pd.read_csv(his_data_path+"/ann_polynomial/train_data.csv") #TODO: use join instead later
        zon_tem_mk = history_data['T_roo'].values[-self.m_prev_steps-1:]  #m+1
        out_tem_mk = history_data['T_oa'].values[-self.m_prev_steps-1:]  # zone air temperature for previos and now, (m+1)*2
        ghi_mk = history_data['GHI'].values[-self.m_prev_steps-1:]  # outdoor air temperature for previos and now, (m+1)*2
        # get energy price in this period
        tims = []
        for i in range(self.m_prev_steps):
            tims.append(tim - self.tau * (i+1))
        tims = tims[::-1]
        tims.append(tim)
        for i in range(self.n_next_steps):
            tims.append(tims[-1] + self.tau)
        ene_pri = self._get_hour_price(tims)
        # current state
        power = history_data['P_tot'].values[-1]
        self.current_state = [tim, zon_tem_mk[-1], out_tem_mk[-1], ghi_mk[-1], power
                         , ene_pri[self.m_prev_steps]]
        # future state
        out_tem_sol_n = self.predictor(self.n_next_steps)  # outdoor air temperature for next, n * 2, temperature first, solar in the end
        self.future_state = list(out_tem_sol_n) + list(ene_pri[-self.n_next_steps:])
        # history state
        self.history_state = list(zon_tem_mk[:self.m_prev_steps]) + list(out_tem_mk[:self.m_prev_steps])
        self.state = tuple(self.current_state + self.future_state + self.history_state)
        # (tim, zon_tem_mk[-1], out_tem_mk[-1], ghi_mk[-1], power
        #               , ene_pri[self.m_prev_steps], out_tem_sol_n)
        assert len(self.state)==self.observation_space.shape[0], "The size of state is not consistent with observation space"

        # self.history = pd.DataFrame()
        # self.history = history_data[['mass_flow', 'T_oa', 'T_roo', 'P_tot','GHI']] # is GHI needed?
        # self.history = self.history_state + self.current_state
        self.action_prev = int(history_data['mass_flow'].values[-1] / 0.55)  # previous one or multipy, can be found in history

        return np.array(self.state, dtype=np.float32), self.action_prev, history_data, {} # why {}

    def render(self, mode='human', close=False):
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        Draws cart-pole with the built-in gym tools.

        :param mode: rendering mode. Read more in Gym docs.
        :param close: flag if rendering procedure should be finished and resources cleaned.
        Used, when environment is closed.
        :return: rendering result
        """
        pass
        
    def close(self):
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.
        :return: True if everything worked out.
        """
        return self.render(close=True)    

    def _get_action_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns action space according to OpenAI Gym API requirements

        :return: Discrete action space of size n, n-levels of mass flowrate from [0,1] with an increment of 1/(n-1)
        """
        self.action_space = spaces.Discrete(self.nActions)
        # return spaces.Discrete(self.nActions)

    def _get_observation_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns observation space according to OpenAI Gym API requirements
        The number of observation number = 6 (current) + n*3 (future) + m*2 (previous)
                 
        :return: Box state space with specified lower and upper bounds for state variables.
        """
        # open gym requires an observation space during initialization
        #TODO why the upper limit of time is 86400 = 24h
        high = np.array([86400.,273.15+35, 273.15+40,1000., 1500., 1.0]+\
                [273.15+40]*self.n_next_steps+[1000.]*self.n_next_steps+[1.]*self.n_next_steps+\
                [273.15+35]*self.m_prev_steps+[1500.]*self.m_prev_steps)
        
        low = np.array([0., 273.15+12, 273.15+(-10.),0., 0., 0.]+\
                [273.15+(-10)]*self.n_next_steps+[0.]*self.n_next_steps+[0.]*self.n_next_steps+\
                [273.15+12]*self.m_prev_steps+[0.]*self.m_prev_steps)
                
        return spaces.Box(low, high)
    
    def _reward_policy(self):
        """
        Internal logic to calculate rewards based on current states.

        :return:, list with 2 elements: energy costs and temperature violations
        """

        # two parts: energy cost + temperature deviations
        # minimization problem: negative
        states = self.state

        # this is a hard-coding. This has to be changed for multi-zones
        power = states[4] 
        time = states[0]
        TZon = states[1] - 273.15 # orginal units from Modelica are SI units
        
        # Here is how the reward should be calculated based on observations
        
        num_zone = 1
        ZTemperature = [TZon] #temperature in C for each zone
        ZPower = [power] # power in W
        # and here we assume even for multizone building, power is given as individual power consumption for each zone, which is an array for multizone model.
        
        
        # temperture upper and lower bound
        T_upper = [30.0 for i in range(24)] # upper bound for unoccuppied: cooling
        T_lower = [12.0 for i in range(24)] # lower bound for unoccuppied: heating 
        T_upper[8:18] = [26.0]*(18-8) # upper bound for occuppied: cooling 
        T_lower[8:18] = [22.0]*(18-8) # lower bound for occuppied: heating
        
        # control period:
        delCtrl = self.tau/3600.0 #may be better to set a variable in initial
        
        #get hour index
        t = int(time)
        t = int((t%86400)/3600) # hour index 0~23

        #calculate penalty for each zone
        overshoot = []
        undershoot = []
        max_violation = []
        penalty = [] #temperature violation for each zone
        cost = [] # erengy cost for each zone

        for k in range(num_zone):
            overshoot.append(max(ZTemperature[k] - T_upper[t] , 0.0))
            undershoot.append(max(T_lower[t] - ZTemperature[k] , 0.0))
            max_violation.append(-overshoot[k] - undershoot[k])
            penalty.append(self.alpha*max_violation[k])
        
        t_pre = int(time-self.tau) if time>self.tau else (time+24*60*60.-self.tau)
        t_pre = int((t_pre%86400)/3600) # hour index 0~23
        
        for k in range(num_zone):
            cost.append(- ZPower[k]/1000. * delCtrl * self.p_g[t_pre])
        
        # calculate action changes: negative as reward
        delta_action = [-abs((self.action_curr - self.action_prev)/(self.nActions - 1.))]
        
        # save cost/penalty for customized use - negative
        self._cost = cost
        self._max_temperature_violation = max_violation
        self._delta_action = delta_action

        # define reward
        if self.rf:
            rewards=self.rf(cost, penalty, delta_action)
        else:
            rewards = np.sum(np.array([cost, penalty, delta_action]))

        # update previous action for next step
        self.action_prev = self.action_curr

        return rewards
        
    def get_state(self, result):
        """
        Extracts the values of model outputs at the end of modeling time interval from simulation result 
        and predicted weather data from future time step

        :return: Values of model outputs as tuple in order specified in `model_outputs` attribute and 
        predicted weather data from existing weather file

        This module is used to override defaulted "get_state" function that 
        only gets states from simulation results.
        """
        # 1. get states that could be measured
        #   model_outputs
        # 2. get states that should be predicted from external predictor
        #   predictor_outputs
        # 3. get states for historical measurement

        model_outputs = self.model_output_names
        
        state_list = [result.final(k) for k in model_outputs]
        
        # get prices at current hour
        time = state_list[0]
        t = int(time)
        t = int((t%86400)/3600) # hour index 0~23
        energy_price = [self.p_g[t]]

        # append price to state list
        state_list += energy_price

        # ============================================
        # get oa predictors for next n_next_steps
        predictor_list = self.predictor(self.n_next_steps)
        # get price for next n_next_steps
        energy_price_next = []
        for i in range(self.n_next_steps):
            time += self.tau 
            t = int(time)
            t = int((t%86400)/3600) # hour index 0~23
            energy_price_next += [self.p_g[t]]
        # append price to predictor list
        predictor_list += energy_price_next

        # =================================================
        # reconstruct time for learning agent
        state_list[0] = int(state_list[0]) % 86400

        # ================================================
        # historical measurement list
        history_list=[]
        if self.m_prev_steps>0:
            TRoo_t = result.final('TRoo')
            pow_t = result.final('PTot')
            TRoo_his= self.history['TRoo']
            pow_his = self.history['PTot']
            TRoo_his.append(TRoo_t)

            pow_his.append(pow_t)

            history_list = TRoo_his[:-1] + pow_his[:-1]
            self.history['TRoo'] = TRoo_his[1:]
            self.history['PTot'] = pow_his[1:]

        return tuple(state_list+predictor_list+history_list) 

    def predictor(self,n):
        """
        Predict weather conditions over a period

        Here we use an ideal weather predictor, which reads data from energyplus weather files

        :param:
            n: number of steps for predicted value

        :return: list, temperature and solar radiance for future n steps

        """
        # temperature and solar in a dataframe
        tem_sol_step = self.read_temperature_solar()
        
        # time = self.state[0]
        # time = self.simulation_start_time
        time = self.current_state[0]
        # return future n steps
        tem = list(tem_sol_step[time+self.tau:time+n*self.tau]['temp_air']+273.15)
        sol = list(tem_sol_step[time+self.tau:time+n*self.tau]['ghi'])

        return tem+sol

    def read_temperature_solar(self):
        """Read temperature and solar radiance from epw file. 
            This module serves as an ideal weather predictor.

        :return: a data frame at an interval of defined time_step
        """
        from pvlib.iotools import read_epw

        file_path = os.path.dirname(os.path.realpath(__file__))
        dat = read_epw(file_path+'/'+self.weather_file)

        tem_sol_h = dat[0][['temp_air','ghi']]
        index_h = np.arange(0,3600.*len(tem_sol_h),3600.)
        tem_sol_h.index = index_h

        # interpolate temperature into simulation steps
        self.tau = 15*60
        index_step = np.arange(0,3600.*len(tem_sol_h),self.tau)

        return interp(tem_sol_h,index_step)



    # # define a method to get sub-step measurements for model outputs. 
    # # The outputs are defined by model_output_names and model_input_names.
    # def get_substep_measurement(self):
    #     """
    #     Get outputs in a smaller step than the control step.
    #     The number of substeps are defined as n_substeps. 
    #     The step-wise simulation results are interpolated into n_substeps.
    #     :return: Tuple of List
    #     """
    #     # following the order as defined in model_output_names
    #     substep_measurement_names = self.model_output_names + self.model_input_names

    #     time = self.result['time'] # get a list of raw time point from modelica simulation results for [t-dt, t].
    #     dt = (time[-1]-time[0])/self.n_substeps
    #     time_intp = np.arange(time[0], time[-1]+dt, dt)

    #     substep_measurement=[]
    #     for var_name in substep_measurement_names:
    #         substep_measurement.append(list(np.interp(time_intp, time, self.result[var_name])))
        
    #     return (substep_measurement_names,substep_measurement)

    # get cost 
    def get_cost(self):
        """Get energy cost and reward afer reward calculation

        :return: a list of cost for multi-zones
        :rtype: List
        """
        return self._cost
    
    # get penalty 
    def get_temperature_violation(self):
        """Get energy cost and reward afer reward calculation

        :return: a list of cost for multi-zones
        :rtype: List
        """
        return self._max_temperature_violation

    # get action changes
    def get_action_changes(self):

        return self._delta_action

def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out

class Net(nn.Module):
    def __init__(self, features):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(features, 10)
        # self.fc2 = nn.Linear(10, 1)
        self.linear_relu1 = nn.Linear(features, 256)
        self.linear_relu2 = nn.Linear(256, 256)
        self.linear_relu3 = nn.Linear(256, 256)
        self.linear_relu4 = nn.Linear(256, 256)
        self.linear_relu5 = nn.Linear(256, 256)
        self.linear_relu6 = nn.Linear(256, 256)

        self.linear7 = nn.Linear(256, 4)
        # self.activation = nn.ReLU()

    def forward(self, x):
        y_pred = self.linear_relu1(x)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu2(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu3(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu4(y_pred)
        y_pred = nn.functional.relu(y_pred)
        
        y_pred = self.linear_relu5(y_pred)
        y_pred = nn.functional.relu(y_pred)
        y_pred = self.linear_relu6(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear7(y_pred)
        
        
        
        # x = self.activation(self.linear_relu1(x))
        # x = self.activation(self.linear_relu2(x))
        # x = self.activation(self.linear_relu3(x))
        # x = self.activation(self.linear_relu4(x))
        # x = self.activation(self.linear_relu5(x))
        # x = self.activation(self.linear6(x))
        
        return y_pred


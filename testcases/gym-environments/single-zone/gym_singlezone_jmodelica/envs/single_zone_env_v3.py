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
from gym.utils import seeding
from modelicagym.environment import FMI2CSEnv, FMI1CSEnv
import os

logger = logging.getLogger(__name__)

class SingleZoneEnv(object):
    """
    Class extracting common logic for JModelica and Dymola environments for CartPole experiments.
    Allows to avoid code duplication.
    Implements all methods for connection to the OpenAI Gym as an environment.

    Description:
        The agent (a fan coil unit system) is controlled to minimize energy cost while maintaining the zone thermal comfort. 
        For any given state the agent may choose to operate the fan at a different speed.
    Reference:
        None
    Observation:
        Type: Box(11)
        Num    Observation                                   Min            Max
        0      Time                                          0              86400
        1      Zone temperature                              273.15 + 12    273.15 + 30
        2      Outdoor temperature                           273.15 + 0     273.15 + 40
        3      Solar radiation                               0              1200
        4      Total power                                   0              1000
        5      Outdoor temperature prediction at next 1 step 273.15 + 0     273.15 + 40
        6      Outdoor temperature prediction at next 2 step 273.15 + 0     273.15 + 40
        7      Outdoor temperature prediction at next 3 step 273.15 + 0     273.15 + 40
        8      Solar radiation prediction at next 1 step     0              1200
        9      Solar radiation prediction at next 2 step     0              1200
        10     Solar radiation prediction at next 3 step     0              1200

    Actions:
        Type: Discret(nActions)
        Num         Action           
        0           Fan off
        ...
        ...
        nAction-1   Fan on at full speed
    Reward:
         Sum of energy costs and zone temperature violations

    """

    # modelicagym API implementation
    def _is_done(self):
        """
        Internal logic that is utilized by parent classes.
        Checks if system states result in a failure

        Note in this design, violations in the system states will not terminate the experiment.
        The experiment should be terminated in the training process by exceeding maximum steps.

        :return: boolean flag if current state of the environment indicates that experiment has ended.

        """
        done = False
        stop_time = self.stop # get current time after do_step
        if stop_time > self.simulation_end_time-self.tau:
            done = True

        return done

    def _get_action_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns action space according to OpenAI Gym API requirements

        :return: Discrete action space of size n, n-levels of mass flowrate from [0,1] with an increment of 1/(n-1)
        """
        return spaces.Discrete(self.nActions)

    def _get_observation_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns observation space according to OpenAI Gym API requirements
        
        The designed observation space for each zone is 
        0. current time stamp
        1. zone temperatures for single or multiple zones; in K
        2. outdoor temperature; in K 
        3. solar radiation
        4. total power
        5-7. outdoor temperature in future 3 steps; in K
        8-10. solar radiation in future 3 steps
            
        :return: Box state space with specified lower and upper bounds for state variables.
        """
        # open gym requires an observation space during initialization

        high = np.array([86400., 273.15+35, 273.15+40,1000., 1500.]+[273.15+40]*self.n_next_steps+[1000.]*self.n_next_steps)
        low = np.array([0., 273.15+12, 273.15+0,0, 0]+[273.15+0]*self.n_next_steps+[0.0]*self.n_next_steps)
        return spaces.Box(low, high)

    # OpenAI Gym API implementation
    def step(self, action):
        """
        OpenAI Gym API. Executes one step in the environment:
        in the current state perform given action to move to the next action.
        Applies force of the defined magnitude in one of two directions, depending on the action parameter sign.

        :param action: alias of an action [0-10] to be performed. 
        :return: next (resulting) state
        """
        # 0 - max flow: 
        #mass_flow_nor = self.mass_flow_nor # norminal flowrate: kg/s 
        action = np.array(action)
        action = [action/float(self.nActions-1)]
        return super(SingleZoneEnv,self).step(action)
    
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
        T_upper[8:18] = [26.0]*(18-8)  # upper bound for occuppied: cooling
        T_lower[8:18] = [22.0]*(18-8)  # lower bound for occuppied: heating
        
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
        
        # save cost/penalty for customized use - negative
        self._cost = cost
        self._max_temperature_violation = max_violation

        # define reward
        if self.rf:
            rewards=self.rf(cost, penalty)
        else:
            rewards=np.sum(np.array([cost, penalty]))

        return rewards

    def do_simulation(self):
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment.
        """
        # part of the step() method extracted for convenience
        # need to re-write this part to add neural network-modelica co-simulation
        logger.debug("Simulation started for time interval {}-{}".format(self.start, self.stop))

        # PyFMI modelling options
        opts = self.model.simulate_options()
        opts['ncp'] = 50
        opts['initialize'] = False

        result = self.model.simulate(start_time=self.start, final_time=self.stop, options=opts)

        return self.get_state(result)
    
    def get_state(self, result):
        """
        Extracts the values of model outputs at the end of modeling time interval from simulation result 
        and predicted weather data from future time step

        :return: Values of model outputs as tuple in order specified in `model_outputs` attribute and 
        predicted weather data from existing weather file
        0. current time stamp                
        1. zone temperatures for single or multiple zones; in K
        2. outdoor temperature; in K 
        3. solar radiation
        4. total power
        5-7. outdoor temperature in future 3 steps; in K
        8-10. solar radiation in future 3 steps

        This module is used to override defaulted "get_state" function that 
        only gets states from simulation results.
        """
        # 1. get states that could be measured
        #   model_outputs
        # 2. get states that should be predicted from external predictor
        #   predictor_outputs
        # 3. get states that comes from neural network TO BE IMPLEMENTED
        #   nn_outputs
        # 4. combine all states TO BE IMPLEMENTED
        #   state = model_outputs + predictor_outputs + nn_outputs
        model_outputs = self.model_output_names
        state_list = [result.final(k) for k in model_outputs]

        predictor_list = self.predictor(self.n_next_steps)

        state_list[0] = int(state_list[0]) % 86400
        return tuple(state_list+predictor_list) 

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
        
        #time = self.state[0]
        time = self.start
        # return future 3 steps
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
        index_step = np.arange(0,3600.*len(tem_sol_h),self.tau)

        return interp(tem_sol_h,index_step)

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

    # define a method to get sub-step measurements for model outputs. 
    # The outputs are defined by model_output_names and model_input_names.
    def get_substep_measurement(self):
        """
        Get outputs in a smaller step than the control step.
        The number of substeps are defined as n_substeps. 
        The step-wise simulation results are interpolated into n_substeps.
        :return: Tuple of List
        """
        # following the order as defined in model_output_names
        substep_measurement_names = self.model_output_names + self.model_input_names

        time = self.result['time'] # get a list of raw time point from modelica simulation results for [t-dt, t].
        dt = (time[-1]-time[0])/self.n_substeps
        time_intp = np.arange(time[0], time[-1]+dt, dt)

        substep_measurement=[]
        for var_name in substep_measurement_names:
            substep_measurement.append(list(np.interp(time_intp, time, self.result[var_name])))
        
        return (substep_measurement_names,substep_measurement)

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

class JModelicaCSSingleZoneEnv(SingleZoneEnv, FMI2CSEnv):
    """
    Wrapper class for creation of cart-pole environment using JModelica-compiled FMU (FMI standard v.2.0).

    Attributes:
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
                 fmu_result_handling='memory',
                 fmu_result_ncp=100,
                 filter_flag=True,
                 alpha=0.01,
                 nActions=11,
                 rf=None,
                 p_g=None,
                 n_substeps=15):

        logger.setLevel(log_level)

        # system parameters
        self.mass_flow_nor = mass_flow_nor 
        self.weather_file = weather_file 
        self.n_next_steps = n_next_steps 

        # virtual environment simulation period
        self.simulation_end_time = simulation_end_time

        # state bounds if any
        
        # experiment parameters
        self.alpha = alpha # Positive: penalty coefficients for temperature violation in reward function 
        self.nActions = nActions # Integer: number of actions for one control variable (level of damper position)

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

        # number of substeps to output
        self.n_substeps=int(n_substeps)

        # others
        self.viewer = None
        self.display = None

        config = {
            'model_input_names': ['uFan'],
            'model_output_names': ['time','TRoo','TOut','GHI','PTot'],
            'model_parameters': {},
            'time_step': time_step,
            'fmu_result_handling':fmu_result_handling,
            'fmu_result_ncp':fmu_result_ncp,
            'filter_flag':filter_flag 
        }

        # initialize some metadata 
        self._cost = []
        self._max_temperature_violation = []
        
        # specify fmu path and model
        fmu_path = os.path.dirname(os.path.realpath(__file__))
        super(JModelicaCSSingleZoneEnv,self).__init__(fmu_path+"/SingleZoneFCU.fmu",
                         config, log_level=log_level,
                         simulation_start_time=simulation_start_time)
       # location of fmu is set to current working directory

def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out
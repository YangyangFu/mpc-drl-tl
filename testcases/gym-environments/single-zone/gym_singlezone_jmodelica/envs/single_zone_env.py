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
from modelicagym.environment import FMI2CSEnv, FMI1CSEnv


logger = logging.getLogger(__name__)

class SingleZoneEnv(object):
    """
    Class extracting common logic for JModelica and Dymola environments for CartPole experiments.
    Allows to avoid code duplication.
    Implements all methods for connection to the OpenAI Gym as an environment.


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

        return done

    def _get_action_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns action space according to OpenAI Gym API requirements

        :return: Discrete action space of size 5, 5-levels of mass flowrate.
        """
        return spaces.Discrete(5)

    def _get_observation_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns observation space according to OpenAI Gym API requirements
        
        The designed observation space for each zone is 
        1. time in second
        2. zone temperatures for single or multiple zones; in K
        3. outdoor temperature; in K 
        4. solar radiation
        5. total power
        6-8. outdoor temperature in future 3 steps; in C
        9-11. solar radiation in future 3 steps
            
        :return: Box state space with specified lower and upper bounds for state variables.
        """
        high = np.array([np.inf, 273.15+35, 273.15+45,np.inf, np.inf,45,45,45,np.inf,np.inf,np.inf])
        low = np.array([np.inf, 273.15-15, 273.15-45,0, 0, -45,-45,-45,0,0,0])
        return spaces.Box(low, high)

    # OpenAI Gym API implementation
    def step(self, action):
        """
        OpenAI Gym API. Executes one step in the environment:
        in the current state perform given action to move to the next action.
        Applies force of the defined magnitude in one of two directions, depending on the action parameter sign.

        :param action: alias of an action [0-4] to be performed. 
        :return: next (resulting) state
        """
        # 0 - max flow: 
        mass_flow_nor = self.mass_flow_nor; # norminal flowrate: kg/s 
        action = action + 1
        action = mass_flow_nor*action/4.
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
        power = states(4) 
        time = states(0)
        TZon = states(1)

        # Here is how the reward should be calculated based on observations

        return [-1,0]


    def get_state(self, result):
        """
        Extracts the values of model outputs at the end of modeling time interval from simulation result 
        and predicted weather data from future time step

        :return: Values of model outputs as tuple in order specified in `model_outputs` attribute and 
        predicted weather data from existing weather file
                    
        1. time in second
        2. zone temperatures for single or multiple zones; in K
        3. outdoor temperature; in K 
        4. solar radiation
        5. total power
        6-8. outdoor temperature in future 3 steps; in C
        9-11. solar radiation in future 3 steps

        This module is used to override defaulted "get_state" function that 
        only gets states from simulation results.
        """
        # 1. get states that could be measured
        #   model_outputs
        # 2. get states that should be predicted from external predictor
        #   predictor_outputs

        model_outputs = self.model_output_names
        state_list = [result.final(k) for k in model_outputs]

        predictor_list = self.predictor(self.npre_step)

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
        tem = list(tem_sol_step[time+self.tau:time+n*self.tau]['temp_air'])
        sol = list(tem_sol_step[time+self.tau:time+n*self.tau]['ghi'])

        return tem+sol

    def read_temperature_solar(self):
        """Read temperature and solar radiance from epw file. 
            This module serves as an ideal weather predictor.

        :return: a data frame at an interval of defined time_step
        """
        from pvlib.iotools import read_epw

        dat = read_epw(self.weather_file)

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


class JModelicaCSSingleZoneEnv(SingleZoneEnv, FMI2CSEnv):
    """
    Wrapper class for creation of cart-pole environment using JModelica-compiled FMU (FMI standard v.2.0).

    Attributes:
        mass_flow_nor (float): List, norminal mass flow rate of VAV terminals.
        time_step (float): time difference between simulation steps.

    """

    def __init__(self,
                 mass_flow_nor,
                 weather_file,
                 npre_step,
                 simulation_start_time,
                 time_step,
                 log_level):

        logger.setLevel(log_level)

        # system parameters
        self.mass_flow_nor = mass_flow_nor
        self.weather_file = weather_file
        self.npre_step = npre_step
        # state bounds if any
        
        # experiment parameters
 
        # others
        self.viewer = None
        self.display = None

        config = {
            'model_input_names': ['uFan'],
            'model_output_names': ['time','TRoo','TOut','GHI','PTot'],
            'model_parameters': {},
            'time_step': time_step
        }

        super(JModelicaCSSingleZoneEnv,self).__init__("./SingleZoneVAV.fmu",
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
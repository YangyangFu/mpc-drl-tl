# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

"""
Classic cart-pole example implemented with an FMU simulating a cart-pole system.
Implementation inspired by OpenAI Gym examples:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

import logging
import math
import numpy as np
from gym import spaces
from modelicagym.environment import FMI2CSEnv, FMI1CSEnv


logger = logging.getLogger(__name__)

NINETY_DEGREES_IN_RAD = (90 / 180) * math.pi
TWELVE_DEGREES_IN_RAD = (12 / 180) * math.pi


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
        Checks if cart position or pole angle are inside required bounds, defined by thresholds:
        x_threshold - 2.4
        angle threshold - 12 degrees

        :return: boolean flag if current state of the environment indicates that experiment has ended.
        True, if cart is not further than 2.4 from the starting point
        and angle of pole deflection from vertical is less than 12 degrees
        """
        if self.stop >= self.episode_length:
            done = True
        else:
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

        :return: Box state space with specified lower and upper bounds for state variables.
        """
        high = np.array([self.x_threshold, np.inf, self.theta_threshold, np.inf])
        return spaces.Box(-high, high)

    # OpenAI Gym API implementation
    def step(self, action):
        """
        OpenAI Gym API. Executes one step in the environment:
        in the current state perform given action to move to the next action.
        Applies force of the defined magnitude in one of two directions, depending on the action parameter sign.

        :param action: alias of an action to be performed. If action > 0 - push to the right, else - push left.
        :return: next (resulting) state
        """
        mass_flow_nor = self.mass_flow_nor; # norminal flowrate 1 kg/s
        action = action + 1
        action = mass_flow_nor*action/5.
        return super(SingleZoneEnv,self).step(action)
    
    def _reward_policy(self):
        pass


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
        positive_reward (int): positive reward for RL agent.
        negative_reward (int): negative reward for RL agent.
    """

    def __init__(self,
                 mass_flow_nor,
                 time_step,
                 log_level):

        logger.setLevel(log_level)

        # system parameters
        self.mass_flow_nor = mass_flow_nor

        # state bounds if any

        # others
        self.viewer = None
        self.display = None


        config = {
            'model_input_names': ['uFan'],
            'model_output_names': ['TRoo', 'CO2Roo', 'powHea', 'powCoo'],
            'model_parameters': {},
            'time_step': time_step
        }

        super(JModelicaCSSingleZoneEnv,self).__init__("./SingleZoneVAV.fmu",
                         config, log_level)
       # location of fmu is set to current working directory

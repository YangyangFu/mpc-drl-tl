import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# initial configuration: can be changed when making the environment
config = {
        'fmu_result_handling':'memory',
        'fmu_result_ncp':100.,
        'filter_flag':True}

register(
    id='JModelicaCSSingleZoneTemperatureEnv-v0',
    entry_point='gym_singlezone_temperature.envs:JModelicaCSSingleZoneTemperatureEnv',
    kwargs = config
)

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# initial configuration: can be changed when making the environment
config = {
        'fmu_result_handling':'memory',
        'fmu_result_ncp':100.,
        'filter_flag':True}

register(
    id='JModelicaCSFiveZoneAirEnv-v0',
    entry_point='gym_fivezoneair_jmodelica.envs.five_zone_air_env_v0:JModelicaCSFiveZoneAirEnv',
    kwargs = config
)

register(
    id='JModelicaCSFiveZoneAirEnv-v1',
    entry_point='gym_fivezoneair_jmodelica.envs.five_zone_air_env_v1:JModelicaCSFiveZoneAirEnv',
    kwargs = config
)

register(
    id='JModelicaCSFiveZoneAirEnv-v2',
    entry_point='gym_fivezoneair_jmodelica.envs.five_zone_air_env_v2:JModelicaCSFiveZoneAirEnv',
    kwargs = config
)
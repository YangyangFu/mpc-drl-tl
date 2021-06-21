import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# initial configuration: can be changed when making the environment
config = {
        'fmu_result_handling':'memory',
        'fmu_result_ncp':100.,
        'filter_flag':True}

register(
    id='JModelicaCSSingleZoneEnv-v0',
    entry_point='gym_singlezone_jmodelica.envs.single_zone_env_v0:JModelicaCSSingleZoneEnv',
    kwargs = config
)

register(
    id='JModelicaCSSingleZoneEnv-v1',
    entry_point='gym_singlezone_jmodelica.envs.single_zone_env_v1:JModelicaCSSingleZoneEnv',
    kwargs = config
)

register(
    id='JModelicaCSSingleZoneEnv-v1.1',
    entry_point='gym_singlezone_jmodelica.envs.single_zone_env_v1.1:JModelicaCSSingleZoneEnv',
    kwargs = config
)

register(
    id='JModelicaCSSingleZoneEnv-v2',
    entry_point='gym_singlezone_jmodelica.envs.single_zone_env_v2:JModelicaCSSingleZoneEnv',
    kwargs = config
)

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# initial configuration: can be changed when making the environment
config = {
        'fmu_result_handling':'memory',
        'fmu_result_ncp':100.,
        'filter_flag':True}

# # example
# register(
#     id='JModelicaCSSingleZoneEnv-action-v2',
#     entry_point='gym_singlezone_jmodelica.envs.single_zone_action_env_v2:JModelicaCSSingleZoneEnv',
#     kwargs=config
# )
# new enviornment
register(
    id='SingleZoneEnv-ANN',
    entry_point='gym_singlezone_rom.envs.single_zone_ann:ANNSingleZoneEnv',
    kwargs=config
)

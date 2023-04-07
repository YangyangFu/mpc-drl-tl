import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# initial configuration: can be changed when making the environment
# config = {
#         'fmu_result_handling':'memory',
#         'fmu_result_ncp':100.,
#         'filter_flag':True}

# # example
# register(
#     id='JModelicaCSSingleZoneEnv-action-v2',
#     entry_point='gym_singlezone_jmodelica.envs.single_zone_action_env_v2:JModelicaCSSingleZoneEnv',
#     kwargs=config
# )
# new enviornment
register(
    id='Example-v1',
    entry_point='env_example.envs.cartpole:CartPoleEnv',
    # kwargs=config
)


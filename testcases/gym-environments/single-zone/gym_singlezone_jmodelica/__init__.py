import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# initial configuration: can be changed when making the environment
config = {}

register(
    id='JModelicaCSSingleZoneEnv-v0',
    entry_point='gym_singlezone_jmodelica.envs:JModelicaCSSingleZoneEnv',
    kwargs = config
)
import pickle
from tianshou.data import ReplayBuffer, Batch
from tianshou.utils.statistics import RunningMeanStd
import argparse
import logging
import os

def initialize_logger():
    """ Initialize a logger for the script. """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """ Load data from a pickle file. """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, "rb") as f:
        return pickle.load(f)

def create_and_populate_buffer(loaded_data):
    """ Create and populate the Replay Buffer. """
    buffer_size = len(loaded_data)
    expert_buffer = ReplayBuffer(buffer_size)

    for data in loaded_data:
        if isinstance(data, Batch):
            expert_buffer.add(data)
        else:
            expert_buffer.add(
                Batch(
                    obs=data['obs'],
                    act=data['act'],
                    rew=data['rew'],
                    done=data['done'],
                    obs_next=data['obs_next'],
                    terminated=data.get('terminated', data['done']),
                    truncated=data.get('truncated', False)
                )
            )
    
    return expert_buffer

def normalize_data(expert_buffer):
    """ Normalize data in the Replay Buffer. """
    rms = RunningMeanStd()

    # Normalize observations
    obs_array = expert_buffer._meta.__dict__['obs']
    rms.update(obs_array)
    expert_buffer._meta.__dict__['obs'] = rms.norm(obs_array)

    # Normalize next observations
    obs_next_array = expert_buffer._meta.__dict__['obs_next']
    rms.update(obs_next_array)
    expert_buffer._meta.__dict__['obs_next'] = rms.norm(obs_next_array)

def save_normalized_data(expert_buffer, save_path):
    """ Save the normalized data to a file. """
    with open(save_path, "wb") as f:
        pickle.dump(expert_buffer, f)
        logging.info(f"Data successfully saved to {save_path}")

def normalize_trajectory(file_path):
    """ Normalize trajectory data in the given file. """
    try:
        loaded_data = load_data(file_path)
        expert_buffer = create_and_populate_buffer(loaded_data)
        normalize_data(expert_buffer)

        save_name = file_path.split('.')[0] + '_normalized.pkl'
        save_normalized_data(expert_buffer, save_name)

    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == '__main__':
    initialize_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mpc-expert-buffer-path', type=str, default='expert_MPC_JModelicaCSSingleZoneEnv-action-v2-PH48.pkl')
    args = parser.parse_args()

    logging.info("Starting normalization process.")
    normalize_trajectory(args.mpc_expert_buffer_path)
    logging.info("Normalization process completed.")

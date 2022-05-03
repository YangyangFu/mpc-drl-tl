import pandas as pd
import re
import os
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import argparse

def find_all_algorithms(root_dir):
    """Find all DRL algorithms in the root_dir"""
    algors = []
    for it in os.scandir(root_dir):
        if it.is_dir():
            # assume the folder follows the convention: algor_seedXXXX
            algor = it.name.split('_')[0]
            if algor not in algors:
                algors.append(algor)
    return algors

def find_all_files(root_dir, algors, pattern):
    """Find all files under root_dir according to relative pattern."""
    sub_dirs = []
    for it in os.scandir(root_dir):
        for algor in algors:
            if it.is_dir() and algor in it.path:
                sub_dirs.append(it.path)

    file_list = []
    for sub_dir in sub_dirs:
        for dirname, _, files in os.walk(sub_dir):
            for f in files:
                absolute_path = os.path.join(dirname, f)
                if re.match(pattern, absolute_path):
                    file_list.append(absolute_path)

    return file_list

def merge_csv(csv_files, algors):
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    csvs_mean_std = pd.DataFrame()
    
    for algor in algors:
        csvs = pd.DataFrame()
        csvs_mean_std_algor = pd.DataFrame()
        for f in csv_files:
            if algor in f:
                csvs = pd.concat([csvs, pd.read_csv(f, index_col=["env_step"], usecols=["env_step","rew"])], axis=1)
        csvs['mean'] = csvs.mean(axis=1)
        csvs['std'] = csvs.std(axis=1)
        columns = [(algor, 'mean'), (algor, 'std')]
        csvs_mean_std_algor = csvs[['mean', 'std']]
        csvs_mean_std_algor.columns = pd.MultiIndex.from_tuples(columns)
        csvs_mean_std = pd.concat([csvs_mean_std, csvs_mean_std_algor], axis=1)
        #csvs[['mean', 'std']].to_csv(os.path.join(root_dir, algor+'_test_rew.csv'))
    return csvs_mean_std

## define some colors 
COLORS = (
    [
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        'red',
        'pink',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, help="Specify current root directory")
    args = parser.parse_args()

    ## load mpc/rbc rewards
    PHs = [4, 8, 16, 32, 48]
    with open('mpc_rewards.json') as f:
        mpc_rewards = json.load(f)
    
    root_dir = args.root_dir
    mpc = mpc_rewards['mpc']['rewards'][0]
    rbc = mpc_rewards['base']['rewards'][0]

    ## read DRL results
    algors = find_all_algorithms(root_dir)
    print(algors)
    drl_rewards_files = find_all_files(root_dir, algors, re.compile(".*test_rew.csv"))
    drl_all = merge_csv(drl_rewards_files, algors)
    print(drl_all.head())

    ## plot rewards
    sns.set_theme() 
    sns.set(style = "darkgrid") #"darkgrid", "whitegrid", "dark", "white", 
    sns.set(font_scale = 1.5)
    sns.color_palette("bright") #"pastel", "muted", "bright"

    # set x ticks
    xticks = [drl_all.index[i] for i in range(0, len(drl_all.index), 100)]
    xticklabels = [int(drl_all.index[i]/672) for i in range(0, len(drl_all.index), 100)]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(drl_all.index, [rbc]*len(drl_all.index), lw=1, c= COLORS[0], label='RBC')
    ax.plot(drl_all.index, [mpc]*len(drl_all.index), lw=1, c= COLORS[1], label='MPC')
    for i, algor in enumerate(algors):
        ax.plot(drl_all.index, drl_all[algor,'mean'], lw=1, c= COLORS[i+2], label=algor.upper())
        ax.fill_between(drl_all.index, 
                        drl_all[algor,'mean']+drl_all[algor,'std'],
                        drl_all[algor,'mean']-drl_all[algor,'std'], 
                        alpha=.4, 
                        fc=COLORS[i+2], 
                        lw=0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rewards')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    
    # SET YLIM
    if "DRL-R2" in root_dir:
        ylim = [-3000, 0]
    elif "DRL-R1" in root_dir:
        ylim = [-500, 0]
    ax.set_ylim(ylim)
    plt.legend(loc=4)
    plt.savefig(os.path.join(root_dir, 'rewards.png'))
    plt.savefig(os.path.join(root_dir, 'rewards.pdf'))

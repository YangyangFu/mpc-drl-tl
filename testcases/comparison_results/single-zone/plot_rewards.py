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
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
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
        'green'
    ]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, help="Specify current root directory")
    args = parser.parse_args()

    root_dir = args.root_dir
    mpc_dir = os.path.dirname(root_dir)

    ## load mpc/rbc rewards
    with open(os.path.join(mpc_dir,'mpc','R2','PH=96','mpc_rewards.json')) as f:
        mpc_rewards = json.load(f)

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
    #xticks = [drl_all.index[i] for i in range(0, len(drl_all.index), 100)]
    #xticklabels = [int(drl_all.index[i]/672) for i in range(0, len(drl_all.index), 100)]
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(drl_all.index, [rbc]*len(drl_all.index),
            lw=1, c=COLORS[0], label='RBC: '+str(round(rbc,2))+'± 0')
    ax.plot(drl_all.index, [mpc]*len(drl_all.index),
            lw=1, c=COLORS[1], label='MPC: '+str(round(mpc,2))+'± 0')
    for i, algor in enumerate(algors):
        drl = drl_all[algor]
        drl.dropna(inplace=True)
        name = algor.upper()
        mean = float(drl['mean'].dropna().iloc[-1])
        std = float(drl['std'].dropna().iloc[-1])
        label = '{name}: {mean} ± {std}'.format(name=name, mean=round(mean,2), std=round(std,2))
        ax.plot(drl.index, drl['mean'], lw=0.5, c=COLORS[i+2], label=label)
        ax.fill_between(drl.index,
                        drl['mean']+drl['std'],
                        drl['mean']-drl['std'], 
                        alpha=.4, 
                        fc=COLORS[i+2], 
                        lw=0)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Rewards')
    #ax.set_xticks(xticks)
    #ax.set_xticklabels(xticklabels)
    
    ylim = [-2000, 0]
    ax.set_ylim(ylim)
    plt.legend(loc=4)
    plt.savefig(os.path.join(root_dir, 'rewards.png'))
    plt.savefig(os.path.join(root_dir, 'rewards.pdf'))

# calculate normalized score
rew_max = drl_all.xs('mean', axis=1, level=1, drop_level=False).max()
rew_max[('mpc','mean')] = mpc
rew_max[('rbc','mean')] = rbc
print(rew_max)

norm_rew_max = (-rew_max/rew_max.min()+2)*100
index = norm_rew_max.index
index = [i[0].upper() for i in index]
norm_rew_max.index = index
norm_rew_max.to_csv('rewards_max.csv')

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(8, 4))
nAlgors = len(index)
plt.barh(range(nAlgors), norm_rew_max, height=1.0, fill=True, ec='k')
plt.yticks(range(nAlgors), index)
plt.xlabel("Normalized score [%]")
plt.savefig('normalized-score.png')
plt.savefig('normalized-score.pdf')

#!/usr/bin/env python3

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
import tqdm
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def find_all_files(root_dir, algor, pattern, task='JModelicaCSSingleZoneEnv-price-v1'):
    """Find all files under root_dir according to relative pattern."""
    sub_dirs=[]
    for it in os.scandir(root_dir):
        if it.is_dir():
            sub_dirs.append(it.path)

    file_list = []
    
    for sub_dir in sub_dirs:
        data_path = os.path.join(sub_dir, 'log_'+algor, task)
        for dirname, _, files in os.walk(data_path):
            for f in files:
                absolute_path = os.path.join(dirname, f)
                if re.match(pattern, absolute_path):
                    file_list.append(absolute_path)

    return file_list

def convert_tfevents_to_csv(root_dir, algor, task, plot_test=True, refresh=False):
    """Recursively convert test/rew from all tfevent file under root_dir to csv.

    This function assumes that there is at most one tfevents file in each directory
    and will add suffix to that directory.

    :param bool refresh: re-create csv file under any condition.
    """
    tfevent_files = find_all_files(root_dir, algor, re.compile(r"^.*tfevents.*$"), task)
    print(f"Converting {len(tfevent_files)} tfevents files under {root_dir} ...")
    result = {}
    outfile_name = 'test_rew.csv' if plot_test else 'train_rew.csv'
    with tqdm.tqdm(tfevent_files) as t:
        for tfevent_file in t:
            t.set_postfix(file=tfevent_file)
            output_file = os.path.join(os.path.split(tfevent_file)[0], outfile_name)
            if os.path.exists(output_file) and not refresh:
                content = list(csv.reader(open(output_file, "r")))
                if content[0] == ["env_step", "rew", "time"]:
                    for i in range(1, len(content)):
                        content[i] = list(map(eval, content[i]))
                    result[output_file] = content
                    continue
            ea = event_accumulator.EventAccumulator(tfevent_file)
            ea.Reload()
            initial_time = ea._first_event_timestamp
            content = [["env_step", "rew", "time"]]
            rewards_tag = "test/reward" if plot_test else "train/reward"
            for test_rew in ea.scalars.Items(rewards_tag):
                content.append(
                    [
                        round(test_rew.step, 4),
                        round(test_rew.value, 4),
                        round(test_rew.wall_time - initial_time, 4),
                    ]
                )
            csv.writer(open(output_file, 'w')).writerows(content)
            result[output_file] = content
    return result

def plot_reward(csv_files, plot_test=True):
    # assume this is only one .csv file
    keys = [key for key in csv_files.keys()]
    plot_tag = "test_" if plot_test else "train_"
    for key in keys:
        dir_name = os.path.dirname(os.path.dirname(os.path.dirname(key)))
        print(dir_name)
        data = pd.read_csv(key)

        plt.figure(figsize=(8, 6))
        plt.plot(data['env_step'], data['rew'])
        plt.grid()
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.savefig(os.path.join(dir_name, plot_tag+"rewards.pdf"))
        plt.savefig(os.path.join(dir_name, plot_tag+"rewards.png"))
        plt.close()

        # move test_rew.csv to dirname
        rew_file = "test_rew.csv" if plot_test else "train_rew.csv"
        des_path = os.path.join(dir_name, rew_file)
        if os.path.exists(des_path):
            os.remove(des_path)
        os.rename(key, des_path)

def plot_final_epoch(root_dir, algor, task):

    sub_dirs = []
    for it in os.scandir(root_dir):
        if it.is_dir():
            sub_dirs.append(it.path)

    for sub_dir in sub_dirs:
        data_path = os.path.join(sub_dir, 'log_'+algor, task)
        acts = np.load(os.path.join(data_path, 'his_act.npy'), allow_pickle=True)
        obss = np.load(os.path.join(data_path, 'his_obs.npy'), allow_pickle=True)
        # get mean and variance for normalized observations
        obs_mean = np.load(os.path.join(data_path, 'obs_mean.npy'), allow_pickle=True) if os.path.exists(os.path.join(data_path, 'obs_mean.npy')) else [0.]*len(obss[0,:])
        obs_var = np.load(os.path.join(data_path, 'obs_var.npy'), allow_pickle=True) if os.path.exists(os.path.join(data_path, 'obs_var.npy')) else [1.]*len(obss[0,:])

        TRoo_obs = [T*np.sqrt(obs_var[1])+obs_mean[1] - 273.15  for T in obss[:, 1]]
        TOut_obs = [T*np.sqrt(obs_var[2])+obs_mean[2] - 273.15 for T in obss[:, 2]]

        # temperture
        t = range(len(TRoo_obs))
        ndays = int(len(TRoo_obs)//96.)
        print("we have " + str(ndays) + " days of data)")
        T_up = 26.0*np.ones([len(TRoo_obs)])
        T_low = 22.0*np.ones([len(TRoo_obs)])

        T_up = [30.0 for i in range(len(TRoo_obs))]
        T_low = [12.0 for i in range(len(TRoo_obs))]
        for i in range(ndays):
            for j in range((18-8)*4):
                T_up[i*24*4 + (j) + 4*8] = 26.0
                T_low[i*24*4 + (j) + 4*8] = 22.0

        # power 
        power_obs = [p*np.sqrt(obs_var[4])+obs_mean[4] for p in obss[:,4]]
        # price
        energy_price = [p*np.sqrt(obs_var[5])+obs_mean[5] for p in obss[:,5]]

        plt.figure(figsize=(12, 12))
        plt.subplot(411)
        plt.plot(t, [energy_price[i] for i in range(len(t))])
        plt.ylabel("Price [$/kWh]")
        #plt.xlabel("Time Step")
        plt.grid()

        plt.subplot(412)
        plt.plot(t, [(acts[i]+1)/2. for i in range(len(t))])
        plt.ylabel("Speed")
        #plt.xlabel("Time Step")
        plt.grid()

        plt.subplot(413)
        plt.plot(t, TOut_obs, 'b', label="Outdoor")
        plt.plot(t, T_up, 'r', t, T_low, 'r')
        plt.plot(t, TRoo_obs, 'k', label="Indoor")
        #plt.ylim([10, 40])
        plt.grid()
        plt.legend()
        plt.ylabel("Temperaure [C]")
        #plt.xlabel("Time Step")

        plt.subplot(414)
        plt.plot(t, [power_obs[i] for i in range(len(t))])
        plt.ylabel("Power [W]")
        plt.xlabel("Time Step")
        plt.grid()

        plt.savefig(os.path.join(sub_dir, "final_epoch.pdf"))
        plt.savefig(os.path.join(sub_dir, "final_epoch.png"))
        plt.close()

        # save perfornance
        energy = sum(power_obs)/4./1000. # kWh
        cost = sum([power_obs[i]*energy_price[i] for i in range(len(power_obs))])/4./1000. # $
        dT_overshoot = [max(TRoo_obs[i]-T_up[i], 0.) for i in range(len(TRoo_obs))]
        dT_undershoot = [max(T_low[i]-TRoo_obs[i], 0.) for i in range(len(TRoo_obs))]
        dTh = (sum(dT_overshoot) + sum(dT_undershoot))/4.
        dT_max = max(max(dT_overshoot), max(dT_undershoot))

        metric = {'energy': energy,
                    'energy_cost': cost,
                    'total_temp_violation': dTh,
                    'max_temp_violation': dT_max}
        with open(os.path.join(sub_dir,'drl_metric.json'), 'w') as outfile:
            json.dump(metric, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--refresh',
        action="store_true",
        help="Re-generate all csv files instead of using existing one."
    )
    parser.add_argument(
        '--remove-zero',
        action="store_true",
        help="Remove the data point of env_step == 0."
    )
    parser.add_argument('--root-dir', type=str)
    parser.add_argument('--algor', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--plot-test', type=int)
    args = parser.parse_args()
    print(args.plot_test)
    plot_test = bool(args.plot_test)
    print(plot_test)
    csv_files = convert_tfevents_to_csv(args.root_dir, args.algor, args.task, plot_test, args.refresh)
    plot_reward(csv_files, plot_test)
    plot_final_epoch(args.root_dir, args.algor, args.task)

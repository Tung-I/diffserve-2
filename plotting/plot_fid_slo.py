import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import ast

model_comb_thres_config = [((2.0, 3.0), 0.2, 0.5, 0.6), ((2.0, 3.0), 0.5, 0.0, 0.5), ((2.0, 3.0), 0.4, 0.0, 0.4), ((2.0, 3.0), 0.3, 0.0, 0.3), ((2.0, 3.0), 0.2, 0.0, 0.2), ((2.0, 3.0), 0.1, 0.0, 0.1), ((1.0, 2.0), 0.8, 0.6, 0.92), ((1.0, 2.0), 0.8, 0.4, 0.88), ((1.0, 2.0), 0.6, 0.6, 0.84), ((1.0, 2.0), 0.8, 0.2, 0.84), ((1.0, 2.0), 0.7, 0.4, 0.82), ((1.0, 2.0), 0.8, 0.1, 0.82), ((1.0, 2.0), 0.6, 0.4, 0.76), ((1.0, 2.0), 0.7, 0.2, 0.76), ((1.0, 2.0), 0.5, 0.4, 0.7), ((1.0, 2.0), 0.6, 0.2, 0.68), ((1.0, 2.0), 0.4, 0.4, 0.64), ((1.0, 2.0), 0.6, 0.1, 0.64), ((1.0, 2.0), 0.5, 0.2, 0.6), ((1.0, 2.0), 0.4, 0.3, 0.58), ((1.0, 2.0), 0.5, 0.1, 0.55), ((1.0, 2.0), 0.4, 0.2, 0.52), ((1.0, 2.0), 0.1, 0.4, 0.46), ((1.0, 2.0), 0.4, 0.1, 0.46), ((1.0, 2.0), 0.3, 0.2, 0.44), ((1.0, 2.0), 0.0, 0.4, 0.4), ((1.0, 2.0), 0.1, 0.3, 0.37), ((1.0, 2.0), 0.2, 0.2, 0.36), ((1.0, 2.0), 0.3, 0.1, 0.37), ((1.0, 2.0), 0.0, 0.3, 0.3), ((1.0, 2.0), 0.1, 0.2, 0.28), ((1.0, 2.0), 0.2, 0.1, 0.28), ((1.0, 2.0), 0.0, 0.2, 0.2), ((1.0, 2.0), 0.1, 0.1, 0.19), ((1.0, 2.0), 0.2, 0.0, 0.2), ((1.0, 2.0), 0.0, 0.1, 0.1), ((1.0, 2.0), 0.1, 0.0, 0.1), ((0.0, 1.0), 0.7, 0.1, 0.73), ((0.0, 1.0), 0.7, 0.0, 0.7), ((0.0, 1.0), 0.6, 0.1, 0.64), ((0.0, 1.0), 0.6, 0.0, 0.6), ((0.0, 1.0), 0.5, 0.1, 0.55), ((0.0, 1.0), 0.5, 0.0, 0.5), ((0.0, 1.0), 0.4, 0.1, 0.46), ((0.0, 1.0), 0.4, 0.0, 0.4), ((0.0, 1.0), 0.3, 0.0, 0.3), ((0.0, 1.0), 0.2, 0.0, 0.2), ((0.0, 1.0), 0.1, 0.0, 0.1), ((0.0, 1.0), 0.0, 0.0, 0.0)]
model_fid_config = [19.29, 19.3, 19.42, 19.52, 19.77, 20.16, 20.5, 20.52, 20.54, 20.55, 20.56, 20.59, 20.62, 20.65, 20.72, 20.79, 20.88, 20.93, 20.98, 21.1, 21.2, 21.26, 21.52, 21.6, 21.68, 21.79, 21.98, 22.03, 22.16, 22.37, 22.45, 22.6, 22.93, 23.18, 23.45, 23.85, 24.3, 24.88, 24.91, 25.0, 25.08, 25.35, 25.56, 25.66, 25.98, 26.54, 27.43, 28.51, 29.75]

def parse_logs(log_dir_path, offset, length=200):
    cas_config_path = f"{log_dir_path}/cascade_config_per_second.csv"
    cascade_config = []
    with open(cas_config_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            raw_models = row[0].strip()
            if raw_models == '':
                models = tuple()
            else:
                models = tuple(ast.literal_eval(raw_models))

            router_thres = float(row[1])
            conf_thres = float(row[2])
            cascade_config.append([models, router_thres, conf_thres])
    cascade_config = cascade_config[offset:offset+length]
    fid_scores = np.array(config2fid(cascade_config))
    fid_scores = np.mean(fid_scores.reshape(int(length/5),5), axis=1)

    slo_results = pd.read_csv(f"{log_dir_path}/slo_timeouts_per_second.csv")
    slo_results = np.array(slo_results.values.tolist()[offset:offset+length])
    slo_timeouts = slo_results[:, [1,2]].reshape(int(length/5), 5, 2)
    slo_timeouts = slo_timeouts.sum(axis=1).sum(axis=1)
    slo_total = np.sum(slo_results[:,3].reshape(int(length/5),5), axis=1)
    # slo_vio_ratios = slo_timeouts / slo_total
    slo_vio_ratios = np.divide(
        slo_timeouts, slo_total,
        out=np.zeros_like(slo_timeouts, dtype=float),
        where=slo_total!=0
    )
    return fid_scores, slo_vio_ratios


def config2fid(cascade_configs):
    fid_scores = []
    for config in cascade_configs:
        for idx, model_thres in enumerate(model_comb_thres_config):
            if config[0] == model_thres[0] and config[1] == model_thres[1] and config[2] == model_thres[2]:
                target_fid = model_fid_config[idx]
                fid_scores.append(target_fid)
                break
    return fid_scores


def smoothResults(rawData, window_size): # moving average over 5 values
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(rawData, kernel, mode='valid')
    return smoothed


if __name__ == "__main__":
    y_limit = (0.0, 0.3)
    saved_dir = 'plots'
    os.makedirs(saved_dir, exist_ok=True)
    exp_result_root = "logs_analysis_Qi"

    fid_1qps, slo_1qps = parse_logs(f"{exp_result_root}/logs_1qps", 1)
    fid_2qps, slo_2qps = parse_logs(f"{exp_result_root}/logs_2qps", 2)
    fid_4qps, slo_4qps = parse_logs(f"{exp_result_root}/logs_4qps", 0)
    fid_8qps, slo_8qps = parse_logs(f"{exp_result_root}/logs_8qps", 3)
    fid_12qps, slo_12qps = parse_logs(f"{exp_result_root}/logs_12qps", 2)
    fid_16qps, slo_16qps = parse_logs(f"{exp_result_root}/logs_16qps", 1)
    fid_18qps, slo_18qps = parse_logs(f"{exp_result_root}/logs_18qps", 3)
    fid_19qps, slo_19qps = parse_logs(f"{exp_result_root}/logs_19qps", 2)
    fid_20qps, slo_20qps = parse_logs(f"{exp_result_root}/logs_20qps", 1)
    fid_24qps, slo_24qps = parse_logs(f"{exp_result_root}/logs_24qps", 3)
    fid_28qps, slo_28qps = parse_logs(f"{exp_result_root}/logs_28qps", 3)
    fid_32qps, slo_32qps = parse_logs(f"{exp_result_root}/logs_32qps", 1)


    smoothed_slo_1qps = smoothResults(slo_1qps, 5)
    smoothed_slo_2qps = smoothResults(slo_2qps, 5)
    smoothed_slo_4qps = smoothResults(slo_4qps, 5)
    smoothed_slo_8qps = smoothResults(slo_8qps, 5)
    smoothed_slo_12qps = smoothResults(slo_12qps, 5)
    smoothed_slo_16qps = smoothResults(slo_16qps, 5)
    smoothed_slo_18qps = smoothResults(slo_18qps, 5)
    smoothed_slo_19qps = smoothResults(slo_19qps, 5)
    smoothed_slo_20qps = smoothResults(slo_20qps, 5)
    smoothed_slo_24qps = smoothResults(slo_24qps, 5)
    smoothed_slo_28qps = smoothResults(slo_28qps, 5)
    smoothed_slo_32qps = smoothResults(slo_32qps, 5)

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_1qps)], smoothed_slo_1qps, label="QPS=1")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_2qps)], smoothed_slo_2qps, label="QPS=2")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_4qps)], smoothed_slo_4qps, label="QPS=4")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_8qps)], smoothed_slo_8qps, label="QPS=8")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_12qps)], smoothed_slo_12qps, label="QPS=12")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_16qps)], smoothed_slo_16qps, label="QPS=16")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_18qps)], smoothed_slo_18qps, label="QPS=18")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_19qps)], smoothed_slo_19qps, label="QPS=19")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_20qps)], smoothed_slo_20qps, label="QPS=20")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_24qps)], smoothed_slo_24qps, label="QPS=24")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_28qps)], smoothed_slo_28qps, label="QPS=28")
    plt.plot(np.arange(0,200,5)[:len(smoothed_slo_32qps)], smoothed_slo_32qps, label="QPS=32")
    plt.ylim(y_limit)
    plt.xlabel("Time (s)")
    plt.ylabel("SLO violation ratio")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{saved_dir}/slo_violation_ratio.png")
    plt.close()


    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(0,200,5)[:len(fid_1qps)], fid_1qps, label="QPS=1")
    plt.plot(np.arange(0,200,5)[:len(fid_2qps)], fid_2qps, label="QPS=2")
    plt.plot(np.arange(0,200,5)[:len(fid_4qps)], fid_4qps, label="QPS=4")

    plt.plot(np.arange(0,200,5)[:len(fid_8qps)], fid_8qps, label="QPS=8")
    plt.plot(np.arange(0,200,5)[:len(fid_12qps)], fid_12qps, label="QPS=12")
    plt.plot(np.arange(0,200,5)[:len(fid_16qps)], fid_16qps, label="QPS=16")
    plt.plot(np.arange(0,200,5)[:len(fid_18qps)], fid_18qps, label="QPS=18")

    plt.plot(np.arange(0,200,5)[:len(fid_19qps)], fid_19qps, label="QPS=19")
    plt.plot(np.arange(0,200,5)[:len(fid_20qps)], fid_20qps, label="QPS=20")
    plt.plot(np.arange(0,200,5)[:len(fid_24qps)], fid_24qps, label="QPS=24")
    plt.plot(np.arange(0,200,5)[:len(fid_28qps)], fid_28qps, label="QPS=28")
    plt.plot(np.arange(0,200,5)[:len(fid_32qps)], fid_32qps, label="QPS=32")

    plt.xlabel("Time (s)")
    plt.ylabel("FID")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{saved_dir}/fid.png")
    plt.close()
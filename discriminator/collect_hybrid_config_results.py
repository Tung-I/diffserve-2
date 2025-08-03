import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from cleanfid import fid
import torch

# target_ratio is the ratio of hard query
def compute_confidence_threshold(score_file, target_ratio):
    scores = np.loadtxt(score_file)
    if scores.ndim > 1:
        scores = scores[:, 1]
    scores = scores.reshape(-1,4).mean(axis=1)
    sorted_scores = np.sort(scores)
    index = max(int(len(sorted_scores) * target_ratio) - 1, 0)
    return sorted_scores[index]

def compute_router_threshold(feat_weight, feat_csv, target_ratio):
    prompt_feats = pd.read_csv(feat_csv)
    feats = prompt_feats.iloc[:, 1:]
    feat_mean = feats.mean()
    feat_std = feats.std()
    norm_feats = (feats - feat_mean) / feat_std
    scores = norm_feats @ feat_weight
    sorted_scores = np.sort(scores)
    index = max(int(len(sorted_scores) * (1-target_ratio)) - 1, 0)
    return sorted_scores[index], (feat_mean, feat_std)

def collect_hybrid_images_config(light_model_idx, heavy_model_idx,
                                 router_thres, disc_thres,
                                 score_file, feat_file, feat_weight,
                                 image_roots, save_to, batch_size=4):
    scores = np.loadtxt(score_file)
    if scores.ndim > 1:
        scores = scores[:, 1]

    prompt_feats = pd.read_csv(feat_file)
    feat_mean, feat_std = prompt_feats.iloc[:, 1:].mean(), prompt_feats.iloc[:, 1:].std()

    light_dir = Path(image_roots[light_model_idx])
    heavy_dir = Path(image_roots[heavy_model_idx])
    save_to_dir = Path(save_to)
    os.makedirs(save_to_dir, exist_ok=True)

    num_l = 0
    for pmpt_idx in tqdm(range(5000), desc=f"Hybrid routing ({light_model_idx}, {heavy_model_idx})"):
        save_to_prompt = save_to_dir / f"prompt_{pmpt_idx}"
        os.makedirs(save_to_prompt, exist_ok=True)

        feat = prompt_feats.iloc[pmpt_idx, 1:]
        feat_norm = (feat - feat_mean) / feat_std
        router_score = feat_norm @ feat_weight

        if router_score >= router_thres:
            src_dir = heavy_dir / f"prompt_{pmpt_idx}"
        else:
            score_offset = pmpt_idx * batch_size
            scores_per_prompt = scores[score_offset:score_offset + batch_size]
            if np.mean(scores_per_prompt) >= disc_thres:
                src_dir = light_dir / f"prompt_{pmpt_idx}"
                num_l += 1
            else:
                src_dir = heavy_dir / f"prompt_{pmpt_idx}"

        for i in range(batch_size):
            src_img = src_dir / f"{i}.jpg"
            dst_img = save_to_prompt / f"{i}.jpg"
            copyfile(src_img, dst_img)

        with open(save_to_dir / "ratio.txt", "w") as f:
            f.write(f"{num_l},{5000},{num_l/5000:.4f}")

def compute_clean_fid(fake_image_dir, real_image_dir, save_file, dataset='imagenet1k'):
    if dataset == 'imagenet1k':
        if not fid.test_stats_exists("imagenet1k_fid_stats", mode="clean"):
            fid.make_custom_stats("imagenet1k_fid_stats", real_image_dir, mode="clean")
        score = fid.compute_fid(
            fake_image_dir,
            dataset_name="imagenet1k_fid_stats",
            mode="clean",
            dataset_split="custom",
            device=torch.device('cuda')
        )
    else:
        raise NotImplementedError

    if os.path.exists(save_file):
        prev_scores = np.loadtxt(save_file)
        if prev_scores.ndim == 0:
            prev_scores = np.array([prev_scores])
        save_scores = np.append(prev_scores, score)
    else:
        save_scores = np.array([score])
    np.savetxt(save_file, save_scores, fmt="%.2f")
    return score

def collect_hybrid_results(score_dir, feat_file, feat_weight,
                           image_roots, real_image_dir,
                           output_csv, thresholds=np.linspace(0, 1, 11)):
    for light_idx in range(len(image_roots)):
        for heavy_idx in range(len(image_roots)):
            if light_idx >= heavy_idx:
                continue
            score_file = os.path.join(score_dir, f"scores_model_{light_idx}.txt")
            for router_ratio in thresholds:
                router_thres, _ = compute_router_threshold(feat_weight, feat_file, router_ratio)
                for disc_ratio in thresholds:
                    disc_thres = compute_confidence_threshold(score_file, disc_ratio)

                    tmp_save_dir = "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/collected_images_hybrid"
                    collect_hybrid_images_config(light_idx, heavy_idx,
                                                 router_thres, disc_thres,
                                                 score_file, feat_file, feat_weight,
                                                 image_roots, tmp_save_dir)

                    latency_light = MODEL_LATENCIES[light_idx]
                    latency_heavy = MODEL_LATENCIES[heavy_idx]
                    total_ratio = router_ratio + (1 - router_ratio) * disc_ratio
                    latency = (1 - total_ratio) * latency_light + \
                                router_ratio * latency_heavy + \
                                (1 - router_ratio) * disc_ratio * (latency_light + latency_heavy)

                    fid_path = f"fid/hybrid_fid_{light_idx}_{heavy_idx}.txt"
                    fid_score = compute_clean_fid(tmp_save_dir, real_image_dir, fid_path)

                    record = {
                        "light_model": light_idx,
                        "heavy_model": heavy_idx,
                        "router_ratio": round(router_ratio, 2),
                        "disc_ratio": round(disc_ratio, 2),
                        "threshold": round(total_ratio, 4),
                        "latency": round(latency, 2),
                        "fid": round(fid_score, 2)
                    }

                    df = pd.DataFrame([record])
                    if not os.path.exists(output_csv):
                        df.to_csv(output_csv, index=False)
                    else:
                        df.to_csv(output_csv, mode='a', index=False, header=False)
                    print(f"Logged hybrid config: {record}")

MODEL_LATENCIES = [0.5, 1.3, 13, 27]  # in seconds

if __name__ == "__main__":
    score_dir = "confidence_scores"
    feat_file = "../router/prompt_features.csv"
    feat_weight = np.array([0.85, 0.32, 0.72, 0.99, 0.76, 0.07, 0.78, 0.01])

    image_roots = [
        "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sdxl-lightning_2",
        "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sd35-large-turbo_4",
        "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sd35-medium_50",
        "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sd35-large_50"
    ]
    real_image_dir = "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/imagenet1k_rescale"
    output_csv = "config_results_hybrid.csv"

    collect_hybrid_results(score_dir, feat_file, feat_weight, image_roots, real_image_dir, output_csv)
    # collect_hybrid_results(score_dir, feat_file, feat_weight, image_roots, real_image_dir, output_csv, thresholds=[0, 0.5, 1])

import argparse
import time
import csv
import pandas as pd
import numpy as np
import torch
import clip
from cleanfid import fid
from pathlib import Path
import os
import random
from PIL import Image
from promptFeature import PromptAnalysisResources


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale_random",
        action="store_true",
        help="whether to do random scaling",
        default=False,
    )
    return parser

def prompt_load(input_prompt, limit_prompts=None):
    prompts = None
    if input_prompt is not None:  # by file
        with open(input_prompt, encoding='ascii', errors='ignore') as f:
            prompts = f.read().splitlines()
        if limit_prompts is not None:
            prompts = prompts[0 : limit_prompts]
    else:  # by txt/list
        pass
    return prompts

def extract_features_to_csv(prompt_list, resources, output_csv="prompt_features.csv"):
    header = [
        "prompt",
        "prompt_length",
        "token_rarity",
        "num_objects",
        "abstractness",
        "attribute_density",
        "spatial_relations",
        "action_verbs",
        "named_entities"
    ]

    with open(output_csv, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for prompt in prompt_list:
            features = resources.extract_features(prompt)
            row = [
                prompt,
                features["prompt_length"],
                features["token_rarity"],
                features["num_objects"],
                features["abstractness"],
                features["attribute_density"],
                features["spatial_relations"],
                features["action_verbs"],
                features["named_entities"]
            ]
            writer.writerow(row)

    print(f"Saved features to {output_csv}")

def compute_feature_means(csv_path):
    df = pd.read_csv(csv_path)
    feature_columns = df.columns[1:]  # skip "prompt"
    means = df[feature_columns].mean()
    stds = df[feature_columns].std()
    return means, stds

def scale_w_feat(feat_weight, threshold, csv_file, save_to, batch_size, light_model, heavy_model, scale_random=False):
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    
    prompt_feats = pd.read_csv(csv_file)
    feature_mean_values, feature_std_values = compute_feature_means(csv_file)
    conf_score_lst = []
    for pmpt_idx in range(len(prompt_feats)):
        feat = prompt_feats.iloc[pmpt_idx, 1:]
        feat_normalized = (feat - feature_mean_values) / feature_std_values
        conf_score = feat_normalized @ feat_weight
        conf_score_lst.append(conf_score)
    conf_score_lst = np.sort(conf_score_lst)

    image_dir_l = Path(f"/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/{light_model}")
    image_dir_h = Path(f"/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/{heavy_model}")
    save_to_dir = Path(save_to)

    num_l = 0
    for pmpt_idx in range(len(prompt_feats)):
        if pmpt_idx % 500 == 0:
            print(f"Processed text prompt ... {pmpt_idx}/{len(prompt_feats)}")
        save_to_prompt = save_to_dir / f"prompt_{pmpt_idx}"
        if not os.path.exists(save_to_prompt):
            os.makedirs(save_to_prompt)

        feat = prompt_feats.iloc[pmpt_idx, 1:]
        feat_normalized = (feat - feature_mean_values) / feature_std_values
        
        conf_score = random.uniform(0,1) if scale_random else feat_normalized @ feat_weight
        conf_thres = threshold if scale_random else conf_score_lst[max([int(len(conf_score_lst)*threshold)-1, 0])]
        
        if conf_score <= conf_thres:
            for i in range(batch_size):
                image = Image.open(f"{image_dir_l}/prompt_{pmpt_idx}/{i}.jpg")
                image.save(f"{save_to_prompt}/{i}.jpg")
            num_l += 1
        else:
            for i in range(batch_size):
                image = Image.open(f"{image_dir_h}/prompt_{pmpt_idx}/{i}.jpg")
                image.save(f"{save_to_prompt}/{i}.jpg")
        np.savetxt(save_to_dir/"ratio.txt", np.array([num_l, pmpt_idx+1, num_l/(pmpt_idx+1)]), fmt="%.2f")

def do_cmp_clean_fid(fake_image_dir, real_image_dir, save_file, dataset='imagenet1k'):
    if dataset == 'imagenet1k':
        if not fid.test_stats_exists("imagenet1k_fid_stats", mode="clean"):
            fid.make_custom_stats("imagenet1k_fid_stats", real_image_dir, mode="clean")
        score = fid.compute_fid(fake_image_dir, dataset_name="imagenet1k_fid_stats", mode="clean", 
                                dataset_split="custom", device=torch.device('cuda'))
    print(f"fid: {score}")
    if os.path.exists(save_file):
        load_score = np.loadtxt(save_file)
        save_score = np.append(load_score, score)
    else:
        save_score = np.array([score])
    np.savetxt(save_file, save_score, fmt="%.2f")

def calc_scores(prompt, images, model, preprocess):
    imgs = [preprocess(img) for img in images]
    batch_imgs = torch.stack(imgs).to('cuda')
    text = clip.tokenize(prompt, truncate=True).to('cuda')
    with torch.no_grad():
        logits_per_image, logits_per_text = model(batch_imgs, text)
    return logits_per_text.squeeze().cpu().tolist()

def compute_clipscore(csv_file, image_dir, save_to):
    model, preprocess = clip.load("ViT-B/32", device='cuda')
    text_prompts = pd.read_csv(csv_file)["prompt"]
    avg_score = 0
    for idx, prompt in enumerate(text_prompts):
        curr_img_dir = f"{image_dir}/prompt_{idx}"
        images = [Image.open(f"{curr_img_dir}/{i}.jpg") for i in range(4)]
        scores = calc_scores(prompt, images, model, preprocess)
        mean_score = np.mean(scores)
        # print(f"Current prompt idx: {idx}, mean clipscore: {mean_score}")
        avg_score += mean_score
    
    avg_score /= len(text_prompts)
    print(f"clip: {avg_score}")
    if os.path.exists(save_to):
        load_scores = np.loadtxt(save_to)
        save_scores = np.append(load_scores, avg_score)
    else:
        save_scores = np.array([avg_score])
    np.savetxt(save_to, save_scores, fmt="%.2f")

def main():
    parser = get_parser()
    opt = parser.parse_args()

    feat_weight = np.array([0.85, 0.32, 0.72, 0.99, 0.76, 0.07, 0.78, 0.01])
    
    csv_file = "prompt_features.csv"
    light_model = "sd35-large-turbo_4"
    heavy_model = "sd35-large_50"
    ps = "random" if opt.scale_random else "feat"
    repeat = 30 if opt.scale_random else 1

    for rep in range(repeat):
        for i, threshold in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            print(f"Current trial: {rep}/{i+1} -----------------------------------")
            images_save_to = f"/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/rule_based_{ps}/{int(threshold*10)}"
            scale_w_feat(feat_weight, threshold, csv_file, images_save_to, batch_size=4, 
                        light_model=light_model, heavy_model=heavy_model, scale_random=opt.scale_random)
        
            real_image_dir = "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/imagenet1k_rescale"
            fid_save_to = f"fid/{ps}_eval_{int(threshold*10)}.txt"
            do_cmp_clean_fid(images_save_to, real_image_dir, fid_save_to, dataset='imagenet1k')
            # clip_save_to = f"clip/{ps}_eval.txt"
            # compute_clipscore(csv_file, images_save_to, clip_save_to)


if __name__ == "__main__":
    main()
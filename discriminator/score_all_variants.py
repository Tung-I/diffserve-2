import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import clip

from model import CLIPDiscriminator  # import your model

def load_discriminator(discriminator_path, device):
    clip_model, _ = clip.load("ViT-B/32", device=device)
    model = CLIPDiscriminator(clip_model.float()).to(device)
    model.load_state_dict(torch.load(discriminator_path, map_location=device))
    model.eval()
    return model

def compute_scores_for_model(model_idx, image_root, score_output_dir, discriminator, preprocess, device):
    model_dir = Path(image_root) 
    scores = []

    for prompt_idx in tqdm(range(5000), desc=f"Scoring Model {model_idx}"):
        for i in range(4):  # 4 images per prompt
            image_path = model_dir / f"prompt_{prompt_idx}" / f"{i}.jpg"
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = discriminator(image_tensor)
                result = torch.softmax(pred, dim=1)[0]
                score = [result[0].item(), result[1].item()]

            scores.append(score)

    output_file = Path(score_output_dir) / f"scores_model_{model_idx}.txt"
    np.savetxt(output_file, np.array(scores))
    print(f"Saved scores for model {model_idx} to {output_file}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    discriminator_path = "CLIP_discriminator.pt"
    fake_image_root_dirs = [
        "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sdxl-lightning_2", 
        "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sd35-large-turbo_4", 
        "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sd35-medium_50", 
        "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sd35-large_50"]
    score_output_dir = "confidence_scores"

    os.makedirs(score_output_dir, exist_ok=True)

    # Load CLIP and discriminator once
    _, preprocess = clip.load("ViT-B/32", device=device)
    discriminator = load_discriminator(discriminator_path, device)

    for model_idx in range(4):
        fake_image_root = fake_image_root_dirs[model_idx]
        compute_scores_for_model(model_idx, fake_image_root, score_output_dir, discriminator, preprocess, device)

if __name__ == "__main__":
    main()

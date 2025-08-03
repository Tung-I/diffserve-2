import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import clip
from dataloader import RealFakeImageDataset
from model import CLIPDiscriminator


# Configuration
REAL_PATH = "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/imagenet1k_rescale"
FAKE_PATHS = ["/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sdxl-lightning_2", 
              "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sd35-large-turbo_4", 
              "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sd35-medium_50", 
              "/scratch3/workspace/qizhengyang_umass_edu-diffserve/imagenet1k/sd35-large_50"]
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-4
SAVE_PATH = "CLIP_discriminator.pt"
MAX_IMAGES_PER_PROMPT = 3


def validate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = RealFakeImageDataset(REAL_PATH, FAKE_PATHS, preprocess, max_images_per_prompt=MAX_IMAGES_PER_PROMPT)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = CLIPDiscriminator(clip_model.float()).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training")
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            avg_loss = total_loss / (total if total != 0 else 1)
            acc = correct / total if total > 0 else 0
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc*100:.2f}%")

        train_acc = correct / total
        val_acc = validate(model, val_loader, device)

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved new best model with acc = {val_acc:.4f}")


if __name__ == "__main__":
    train_model()

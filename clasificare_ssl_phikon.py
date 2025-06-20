# -*- coding: utf-8 -*-
!pip install torch torchvision transformers==4.45.1 datasets evaluate scikit-learn matplotlib seaborn --quiet
import os
import shutil
import zipfile
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
ZIP_PATH = "/content/archive.zip"
DATASET_PATH = "/content/data/histology"
SRC_PATH = f"{DATASET_PATH}/archive/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000"
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall("/content/data/")
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DATASET_PATH, split), exist_ok=True)
for category in os.listdir(SRC_PATH):
    category_path = os.path.join(SRC_PATH, category)
    if not os.path.isdir(category_path):
        continue
    images = os.listdir(category_path)
    random.shuffle(images)
    images = images[:int(0.5 * len(images))]
    split_1 = int(0.7 * len(images))
    split_2 = int(0.8 * len(images))
    train_images = images[:split_1]
    val_images = images[split_1:split_2]
    test_images = images[split_2:]
    for split, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        split_path = os.path.join(DATASET_PATH, split, category)
        os.makedirs(split_path, exist_ok=True)
        for img in split_images:
            shutil.move(os.path.join(category_path, img), os.path.join(split_path, img))
for split in ["train", "val", "test"]:
    split_path = os.path.join(DATASET_PATH, split)
    print(f"{split}: {len(os.listdir(split_path))} clase")
    for category in os.listdir(split_path):
        print(f"{category}: {len(os.listdir(os.path.join(split_path, category)))} imagini")

processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}

train_dataset_raw = ImageFolder(root=f"{DATASET_PATH}/train")
val_dataset_raw = ImageFolder(root=f"{DATASET_PATH}/val")
test_dataset_raw = ImageFolder(root=f"{DATASET_PATH}/test")

train_dataset = CustomDataset(train_dataset_raw, transform=train_transforms)
val_dataset = CustomDataset(val_dataset_raw, transform=val_transforms)
test_dataset = CustomDataset(test_dataset_raw, transform=val_transforms)

NUM_CLASSES = len(train_dataset_raw.classes)

model = AutoModelForImageClassification.from_pretrained(
    "owkin/phikon-v2",
    num_labels=NUM_CLASSES
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "precision": precision.compute(predictions=preds, references=labels, average="macro")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="macro")["recall"]
    }

training_args = TrainingArguments(
    output_dir="./phikon-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,
    compute_metrics=compute_metrics
)

trainer.train()

test_preds = trainer.predict(test_dataset)
y_true = test_preds.label_ids
y_pred = np.argmax(test_preds.predictions, axis=1)

print("\n Clasificare completă:\n")
print(classification_report(y_true, y_pred, target_names=test_dataset_raw.classes))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset_raw.classes)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Matrice de confuzie pe test set")
plt.show()

from torchvision import transforms
from module_02_augmentation import get_transform_with_hsv, RemoveBottomTransform
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

import random
import numpy as np
import torch

# Set the seed for reproducibility
def set_seed(seed=19):
    """
    Sets seeds for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Mappings for colors
color_mapping = {
    "red": 0,
    "black": 1,
    "yellow": 2,
    "orange": 3,
    "green": 4,
    "blue": 5,
    "white": 6
}
inverse_color_mapping = {v: k for k, v in color_mapping.items()}

class JerseyDataset(Dataset):
    """
    Each image in 'df' is repeated 'augment_times' times in the dataset length.
    For each sample, with probability 'augment_probability' we apply
    the color+weather transform pipeline. Otherwise, we use a simpler
    pipeline (e.g. just resizing + normalization).
    """

    def __init__(self, df,
                 augment_times=5,
                 augment_probability=0.5,
                 weather_probability=0.5,
                 train=True):
        """
        Args:
            df (pd.DataFrame): Must have columns ["path", "color", "h", "s", "v", "label_id"].
            augment_times (int): How many times each image appears in the dataset (for repeated augmentation).
            augment_probability (float): Probability to apply the heavy HSV+Weather augmentation per sample.
            weather_probability (float): Probability INSIDE the heavy transform that we apply one weather effect.
            train (bool): Whether to apply random spatial transforms (e.g. random crop/flip).
        """
        self.df = df.reset_index(drop=True)
        self.augment_times = augment_times
        self.augment_probability = augment_probability
        self.weather_probability = weather_probability
        self.train = train

    def __len__(self):
        # If we do 'augment_times=5', the dataset has 5x more "samples" than images.
        return len(self.df) * self.augment_times

    def __getitem__(self, idx):
        # 1) Map the artificial "samples" index to an actual image index
        true_idx = idx // self.augment_times
        row = self.df.iloc[true_idx]

        img_path = row["path"]
        color = row["color"]
        label_id = row["label_id"]
        orig_h = row["h"]
        orig_s = row["s"]
        orig_v = row["v"]

        # 2) Load image
        image = Image.open(img_path).convert("RGB")

        # 3) Decide if we apply the heavy transform or not
        if random.random() < self.augment_probability:
            # Heavy transform: color shift (HSV) + optional weather
            transform = get_transform_with_hsv(
                color=color,
                orig_h=orig_h,
                orig_s=orig_s,
                orig_v=orig_v,
                weather_probability=self.weather_probability,  # e.g. 0.5
                train=self.train
            )
        else:
            # Simpler pipeline (no HSV shift, no weather)
            # Possibly some basic transforms if we are in training, or none for val
            if self.train:
                transform = transforms.Compose([
                    # Example training transforms
                    RemoveBottomTransform(crop_ratio=0.40, probability=0.50),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=(-45, 45)),
                    transforms.Resize((224, 224)), #224x224
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])

        # 4) Apply transform
        image = transform(image)

        # 5) Create label
        label = torch.tensor(label_id, dtype=torch.long)

        return image, label, img_path

def save_training_params(output_dir, params):
    """
    Saves training parameters to a text file.

    Args:
        output_dir (str): Directory to save the parameters.
        params (dict): Dictionary of training parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    params_path = os.path.join(output_dir, "training_parameters.txt")

    with open(params_path, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Training parameters saved to {params_path}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_best_model(model, path):
    """
    Saves the best model to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): Path to save the model.
    """
    torch.save(model.state_dict(), path)

def save_images(image_list, output_path, fold_dir, correct=True):
    """
    Saves classified images and their metadata to an Excel file and a directory.

    Args:
        image_list (list): List of image paths, predicted labels, and actual labels.
        output_path (str): Path to save the Excel file.
        fold_dir (str): Directory to save images.
        correct (bool): Indicates whether the images are correctly classified.
    """
    if not image_list:
        print(f"No {'correctly' if correct else 'incorrectly'} classified images to save.")
        return

    label = "correct" if correct else "wrong"
    df = pd.DataFrame(image_list, columns=["path", "pred", "actual"])
    df["pred_color"] = df["pred"].map(inverse_color_mapping)
    df["actual_color"] = df["actual"].map(inverse_color_mapping)
    df = df[["path", "pred_color", "actual_color"]]

    # Prepare directory structure
    base_predicted_dir = os.path.join(fold_dir, "predicted_images")
    os.makedirs(base_predicted_dir, exist_ok=True)

    dir_path = os.path.join(base_predicted_dir, f"{label}_predicted_images")
    os.makedirs(dir_path, exist_ok=True)

    # Save Excel file
    excel_filename = f"{label}_classified.xlsx"
    excel_path = os.path.join(base_predicted_dir, excel_filename)
    df.to_excel(excel_path, index=False)

    # Save images
    for path, pred, actual in image_list:
        try:
            img = Image.open(path)
            file_name = os.path.basename(path)
            pred_color = inverse_color_mapping[pred]
            actual_color = inverse_color_mapping[actual]
            save_path = os.path.join(
                dir_path, f"{file_name}_pred_{pred_color}_act_{actual_color}.jpg"
            )
            img.save(save_path)
        except Exception as e:
            print(f"Error saving image {path}: {e}")

def plot_metrics(train_metrics, val_metrics, metric_name, fold_dir):
    """
    Creates and saves plots for training and validation metrics.

    Args:
        train_metrics (list): List of training metric values.
        val_metrics (list): List of validation metric values.
        metric_name (str): Name of the metric.
        fold_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_metrics) + 1), train_metrics, label=f'Train {metric_name}')
    plt.plot(range(1, len(val_metrics) + 1), val_metrics, label=f'Val {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Epoch')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(fold_dir, f'{metric_name.lower()}_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_path, fold_num=None):
    """
    Plots and saves the confusion matrix.

    Args:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.
        classes (list): List of class names.
        output_path (str): Path to save the confusion matrix.
        fold_num (int, optional): Fold number for display title. Defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')

    title = f"Confusion Matrix"
    if fold_num is not None:
        title += f" (Fold {fold_num})"
    plt.title(title)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

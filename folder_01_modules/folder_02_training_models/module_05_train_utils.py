import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from module_03_helper_functions import (
    save_images,
    plot_metrics,
    inverse_color_mapping,
)

# 1) Train function for each epoch
def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Performs training for one epoch.
    Args:
        model: The model to train.
        train_loader: DataLoader for the training dataset.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device: Device to perform computation (CPU or GPU).
    Returns:
        avg_loss: Average training loss for the epoch.
        accuracy: Training accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# 2) Validation function for each epoch
def validate_epoch(model, val_loader, criterion, device):
    """
    Validates the model for one epoch.
    Args:
        model: The model to validate.
        val_loader: DataLoader for the validation dataset.
        criterion: Loss function.
        device: Device to perform computation (CPU or GPU).
    Returns:
        avg_loss: Average validation loss for the epoch.
        accuracy: Validation accuracy for the epoch.
        all_preds: List of all predicted labels.
        all_labels: List of all actual labels.
        misclassified: List of tuples (path, predicted label, actual label) for misclassified samples.
        correct_samples: List of tuples (path, predicted label, actual label) for correctly classified samples.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    misclassified = []
    correct_samples = []

    with torch.no_grad():
        for images, labels, paths in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((paths[i], preds[i].item(), labels[i].item()))
                else:
                    correct_samples.append((paths[i], preds[i].item(), labels[i].item()))

    avg_loss = val_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, all_preds, all_labels, misclassified, correct_samples

# 3) Grad-CAM
def generate_grad_cam(model, samples, device, output_dir, subfolder_name="grad_cam", model_name="lenet"):
    """
    Generates Grad-CAM visualizations for given samples and saves them to output_dir/subfolder_name.
    Args:
        model: The model to generate Grad-CAMs for.
        samples: List of tuples (path, predicted label, actual label).
        device: Device to perform computation (CPU or GPU).
        output_dir: Directory to save the Grad-CAM images.
        subfolder_name: Subfolder name for Grad-CAM images.
        model_name: Name of the model (used to select target layers).
    """
    grad_cam_dir = os.path.join(output_dir, subfolder_name)
    os.makedirs(grad_cam_dir, exist_ok=True)

    # Select the appropriate target layer based on the model type
    if model_name.lower() == "lenet":
        target_layers = [model.conv_layers[-1]]
    elif model_name.lower() == "vgg11":
        target_layers = [model.features[-2]]
    elif model_name.lower() == "resnet10":
        target_layers = [model.layer4[-1]]
    elif "deit" in model_name.lower():
        print("Skipping Grad-CAM for DeiT models.")
        return

    else:
        raise ValueError(f"Unknown model_name {model_name} for Grad-CAM")

    cam = GradCAM(model=model, target_layers=target_layers)

    for path, pred, actual in samples:
        pil_img = Image.open(path).convert("RGB").resize((224, 224)) #224,224
        image_tensor = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)

        # Target is the actual label to understand why the prediction should have been correct
        targets = [ClassifierOutputTarget(actual)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Extract CAM for the first input

        # Convert tensor -> numpy -> overlay
        rgb_image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        save_path = os.path.join(
            grad_cam_dir,
            f"{os.path.basename(path).split('.')[0]}_pred_{inverse_color_mapping[pred]}_act_{inverse_color_mapping[actual]}.jpg"
        )
        Image.fromarray(cam_image).save(save_path)

# 4) Training logs / plots
def log_training_results(fold_dir, train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Saves training and validation metrics as plots.
    Args:
        fold_dir: Directory to save the plots.
        train_losses: List of training losses.
        val_losses: List of validation losses.
        train_accuracies: List of training accuracies.
        val_accuracies: List of validation accuracies.
    """
    os.makedirs(fold_dir, exist_ok=True)
    plot_metrics(train_losses, val_losses, "Loss", fold_dir)
    plot_metrics(train_accuracies, val_accuracies, "Accuracy", fold_dir)

# 5) Save misclassified and correctly classified images
def save_classified_images(misclassified, correct, fold_dir):
    """
    Saves dataframes and images of misclassified and correctly classified samples in respective folders.
    Args:
        misclassified: List of misclassified samples.
        correct: List of correctly classified samples.
        fold_dir: Directory to save the images and dataframes.
    """
    save_images(
        misclassified,
        os.path.join(fold_dir, "predicted_images/misclassified.xlsx"),
        fold_dir,
        correct=False
    )
    save_images(
        correct,
        os.path.join(fold_dir, "predicted_images/correct_classified.xlsx"),
        fold_dir,
        correct=True
    )

import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import platform
import multiprocessing
from torch.optim.lr_scheduler import StepLR
import time

from module_01_create_train_df import create_dataframe_with_folds
from module_03_helper_functions import (
    set_seed,
    JerseyDataset,
    color_mapping,
    inverse_color_mapping,
    save_best_model,
    plot_confusion_matrix
)
from module_05_train_utils import (
    train_epoch,
    validate_epoch,
    generate_grad_cam,
    log_training_results,
    save_classified_images,
)
from module_04_models import initialize_model


def train_model(
        model_name="lenet",
        learning_rate=1e-3,
        batch_size=16,
        epochs=1,
        device=None,
        output_dir="results",
        num_workers=0,
        persistent_workers=False,
        augment_times=12,
        augment_probability=0.8,
        seed=19,
        max_grad_cam_images=300,
        step_size=3,
        gamma=0.9,
        use_batchnorm=False,
        n_folds=3,
        save_model=True,          # controls whether to save .pth files or not
        save_images=True,
        patience=None
):
    """
    Trains the model using 'n_folds' cross-validation and returns the average validation loss,
    average validation accuracy and averaged epoch metrics (averaged over folds).

    Returns:
        tuple: (avg_val_loss, avg_val_acc, epoch_metrics)
            where epoch_metrics is a dict with keys:
              - "train_losses": list of average training loss per epoch,
              - "val_losses": list of average validation loss per epoch,
              - "train_accuracies": list of average training accuracy per epoch,
              - "val_accuracies": list of average validation accuracy per epoch.
    """
    start_time = time.time()

    # Decide device if not explicitly provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print device info
    if device.type == "cuda":
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print(f"CPU: {platform.processor()}")
        print(f"Number of CPU cores: {multiprocessing.cpu_count()}")

    # Set random seed for reproducibility
    set_seed(seed)

    # Load the DataFrame with fold assignments
    df_train = create_dataframe_with_folds()
    df_train["label_id"] = df_train["color"].map(color_mapping)
    num_classes = len(color_mapping)

    # Create main output directories
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_results_dir = os.path.join(model_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_results_dir, exist_ok=True)
    models_dir = os.path.join(run_results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"[INFO] Training {model_name} - Results saved in: {run_results_dir}")

    # Save training parameters to a text file
    params_txt_path = os.path.join(run_results_dir, "training_params.txt")
    with open(params_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Using device: {device}\n")
        if device.type == "cuda":
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Number of GPUs available: {torch.cuda.device_count()}\n")
            f.write(f"Current CUDA device: {torch.cuda.current_device()}\n")
        else:
            f.write(f"CPU: {platform.processor()}\n")
            f.write(f"Number of CPU cores: {multiprocessing.cpu_count()}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"augment_times: {augment_times}\n")
        f.write(f"augment_probability: {augment_probability}\n")
        f.write(f"step_size: {step_size}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"use_batchnorm: {use_batchnorm}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"n_folds: {n_folds}\n")
        f.write(f"save_model: {save_model}\n")
        f.write(f"save_images: {save_images}\n")
    print(f"Training parameters saved to {params_txt_path}")

    # Prepare lists to store best metrics per fold
    fold_losses = []
    fold_accuracies = []
    fold_epoch_metrics = []  # Hier werden epochbezogene Metriken pro Fold gespeichert

    best_fold_index = -1
    best_fold_val_loss = float('inf')
    best_fold_val_acc = -float('inf')

    # Loop over folds
    for fold in range(n_folds):
        print(f"\n===== Fold {fold + 1} / {n_folds} =====")
        fold_dir = os.path.join(run_results_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        log_path = os.path.join(fold_dir, "training_log.txt")
        log_file = open(log_path, "w", encoding="utf-8")

        # Prepare train/validation split
        train_df = df_train[df_train["fold"] != fold].reset_index(drop=True)
        val_df = df_train[df_train["fold"] == fold].reset_index(drop=True)
        train_games = sorted(train_df["color_and_game"].unique())
        val_games = sorted(val_df["color_and_game"].unique())
        print(f"Train Games ({len(train_games)}): {', '.join(train_games)}")
        print(f"Validation Games ({len(val_games)}): {', '.join(val_games)}")
        log_file.write(f"Fold {fold + 1}:\n")
        log_file.write(f"Train Games ({len(train_games)}): {', '.join(train_games)}\n")
        log_file.write(f"Validation Games ({len(val_games)}): {', '.join(val_games)}\n\n")

        # Create datasets
        train_dataset = JerseyDataset(train_df, augment_times=augment_times, augment_probability=augment_probability)
        val_dataset = JerseyDataset(val_df, augment_times=1, augment_probability=0.0)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, persistent_workers=persistent_workers)

        # Initialize model
        model = initialize_model(model_name, num_classes, device=device, use_batchnorm=use_batchnorm)

        # Write model details if not exists
        model_details_path = os.path.join(run_results_dir, "model_details.txt")
        if not os.path.exists(model_details_path):
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            with open(model_details_path, "w", encoding="utf-8") as details_file:
                details_file.write(str(model) + "\n")
                details_file.write(f"Number of trainable parameters: {trainable_params}\n")
            print(f"[INFO] Model details saved to {model_details_path}")
        else:
            print("[INFO] Model details file already exists, skipping...")

        # Set up optimizer, scheduler, loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_preds = []
        best_labels = []
        best_misclassified = []
        best_correct = []

        # Listen für epoch metrics
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_train_accuracies = []
        epoch_val_accuracies = []

        epochs_no_improvement = 0

        # Epoch loop
        for epoch in range(epochs):
            print(f"[Fold {fold + 1}] Epoch {epoch + 1}/{epochs}")
            log_file.write(f"Epoch {epoch + 1}/{epochs}\n")

            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, all_preds, all_labels, misclassified, correct_samples = validate_epoch(
                model, val_loader, criterion, device
            )

            epoch_train_losses.append(train_loss)
            epoch_val_losses.append(val_loss)
            epoch_train_accuracies.append(train_acc)
            epoch_val_accuracies.append(val_acc)

            msg = (f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")
            print(msg.strip())
            log_file.write(msg)

            scheduler.step()

            # Update best metrics for this fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_preds = all_preds
                best_labels = all_labels
                best_misclassified = misclassified
                best_correct = correct_samples
                if save_model:
                    best_model_path = os.path.join(models_dir, f"{model_name}_fold_{fold + 1}.pth")
                    save_best_model(model, best_model_path)
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
                if patience is not None and epochs_no_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs without improvement in Val Loss.")
                    break

        log_file.close()
        fold_losses.append(best_val_loss)
        fold_accuracies.append(best_val_acc)
        fold_epoch_metrics.append({
            "train_losses": epoch_train_losses,
            "val_losses": epoch_val_losses,
            "train_accuracies": epoch_train_accuracies,
            "val_accuracies": epoch_val_accuracies
        })

        if best_val_loss < best_fold_val_loss:
            best_fold_val_loss = best_val_loss
            best_fold_val_acc = best_val_acc
            best_fold_index = fold

        # Classification Report, Confusion Matrix, Grad-CAM etc. (unverändert)
        target_names = [inverse_color_mapping[i] for i in range(num_classes)]
        report = classification_report(best_labels, best_preds, target_names=target_names, digits=2, zero_division=0)
        report_path = os.path.join(fold_dir, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write("Classification Report\n\n" + report)
        print(f"Classification report saved to {report_path}")

        confusion_matrix_path = os.path.join(fold_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_true=best_labels, y_pred=best_preds, classes=target_names,
                              output_path=confusion_matrix_path, fold_num=fold + 1)
        print(f"Confusion matrix saved to {confusion_matrix_path}")

        if save_images:
            save_classified_images(best_misclassified, best_correct, fold_dir)

        if save_images:
            random.shuffle(best_misclassified)
            random.shuffle(best_correct)
            best_misclassified = best_misclassified[:max_grad_cam_images]
            best_correct = best_correct[:max_grad_cam_images]
            generate_grad_cam(model, best_misclassified, device=device, output_dir=fold_dir,
                              subfolder_name="grad_cam_false", model_name=model_name)
            generate_grad_cam(model, best_correct, device=device, output_dir=fold_dir,
                              subfolder_name="grad_cam_true", model_name=model_name)

        log_training_results(fold_dir, epoch_train_losses, epoch_val_losses, epoch_train_accuracies, epoch_val_accuracies)

        if n_folds == 1:
            break

    avg_val_loss = sum(fold_losses) / len(fold_losses)
    avg_val_acc = sum(fold_accuracies) / len(fold_accuracies)
    results_file = os.path.join(run_results_dir, "final_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        fold_str = " | ".join([f"Loss: {loss:.4f}, Acc: {acc:.2f}%" for loss, acc in zip(fold_losses, fold_accuracies)])
        f.write(f"{model_name} | {fold_str} | Average Validation Loss: {avg_val_loss:.4f} | Average Validation Accuracy: {avg_val_acc:.2f}%\n")
    print("\n=== Training & Evaluation complete! ===")
    print(f"Fold Validation Losses: {fold_losses}")
    print(f"Fold Validation Accuracies: {fold_accuracies}")
    print(f"Average Validation Loss (Folds): {avg_val_loss:.4f}")
    print(f"Average Validation Accuracy (Folds): {avg_val_acc:.2f}%")
    print(f"Final results saved to: {results_file}")

    best_model_info_path = os.path.join(run_results_dir, "best_model_info.txt")
    with open(best_model_info_path, "w", encoding="utf-8") as f:
        f.write("== BEST FOLD INFO ==\n")
        f.write(f"Fold index: {best_fold_index + 1}\n")
        f.write(f"Validation Loss on best fold: {best_fold_val_loss:.4f}\n")
        f.write(f"Validation Accuracy on best fold: {best_fold_val_acc:.2f}%\n")
        if save_model:
            best_model_path = os.path.join(models_dir, f"{model_name}_fold_{best_fold_index + 1}.pth")
            f.write(f"Model file path: {best_model_path}\n")
        else:
            f.write("Model was not saved (save_model=False)\n")
    print(f"Best fold info saved to {best_model_info_path}")

    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours, rem = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    with open(params_txt_path, "a", encoding="utf-8") as f:
        f.write(f"Total training time: {int(hours)}h:{int(minutes)}m:{seconds:.2f}s\n")
    print(f"Total training time: {int(hours)}h:{int(minutes)}m:{seconds:.2f}s")

    # --- Aggregation der epochbezogenen Metriken über alle Folds ---
    min_epochs = min(len(metrics["train_losses"]) for metrics in fold_epoch_metrics)
    avg_epoch_train_losses = [
        sum(metrics["train_losses"][i] for metrics in fold_epoch_metrics) / len(fold_epoch_metrics)
        for i in range(min_epochs)
    ]
    avg_epoch_val_losses = [
        sum(metrics["val_losses"][i] for metrics in fold_epoch_metrics) / len(fold_epoch_metrics)
        for i in range(min_epochs)
    ]
    avg_epoch_train_accuracies = [
        sum(metrics["train_accuracies"][i] for metrics in fold_epoch_metrics) / len(fold_epoch_metrics)
        for i in range(min_epochs)
    ]
    avg_epoch_val_accuracies = [
        sum(metrics["val_accuracies"][i] for metrics in fold_epoch_metrics) / len(fold_epoch_metrics)
        for i in range(min_epochs)
    ]

    epoch_metrics = {
        "train_losses": avg_epoch_train_losses,
        "val_losses": avg_epoch_val_losses,
        "train_accuracies": avg_epoch_train_accuracies,
        "val_accuracies": avg_epoch_val_accuracies
    }

    # Return the average validation loss, average validation accuracy and epoch metrics
    return avg_val_loss, avg_val_acc, epoch_metrics

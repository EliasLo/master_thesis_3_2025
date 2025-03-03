import optuna
import os
import torch
from module_06_train_model import train_model
import optuna.visualization as vis
import matplotlib.pyplot as plt
import pandas as pd

def objective_with_augment_loss(trial, model_name, output_dir):
    # Optimierung mit Heavy-Augmentation: Basisparameter plus Augmentationsparameter
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    use_batchnorm = trial.suggest_categorical("use_batchnorm", [False, True])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    gamma = trial.suggest_float("gamma", 0.6, 1.0)
    augment_times = trial.suggest_categorical("augment_times", [3, 5, 8])
    # Wir fixieren hier augment_probability, z. B. auf 0.5.
    augment_probability = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avg_val_loss, avg_val_acc = train_model(
        model_name=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=5,  # Mehr Epochen wegen Augmentation
        device=device,
        output_dir=output_dir,
        num_workers=0,
        persistent_workers=False,
        augment_probability=augment_probability,
        augment_times=augment_times,
        max_grad_cam_images=0,
        step_size=3,
        gamma=gamma,
        n_folds=3,
        save_model=False,
        save_images=False
    )
    trial.set_user_attr("validation_accuracy", avg_val_acc)
    return avg_val_loss

def run_optuna_with_augment_loss(model_name="lenet", n_trials=10, output_dir="optuna_aug_results"):
    study = optuna.create_study(direction="minimize")

    def objective_wrapper(trial):
        return objective_with_augment_loss(trial, model_name, output_dir)

    study.optimize(objective_wrapper, n_trials=n_trials)

    save_optuna_results_to_excel(study, output_dir, model_name, heavy_augment=True)
    return study

def save_optuna_results_to_excel(study, output_dir, model_name, heavy_augment=False):
    """
    Speichert alle Trials in einer Excel-Datei mit den Spalten:
    Trial | Learning_rate | Batch_size | Gamma | Val_accuracy | Val_loss
    (Bei heavy-augmentation zusätzlich: Augment_times)
    """
    results = []
    for trial in study.trials:
        trial_dict = {
            "Trial": trial.number,
            "Learning_rate": trial.params.get("learning_rate"),
            "Batch_size": trial.params.get("batch_size"),
            "Gamma": trial.params.get("gamma"),
            "Val_loss": trial.value,
            "Val_accuracy": trial.user_attrs.get("validation_accuracy")
        }
        if heavy_augment:
            trial_dict["Augment_times"] = trial.params.get("augment_times")
        results.append(trial_dict)

    df = pd.DataFrame(results)
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    excel_path = os.path.join(model_dir, "optuna_results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Optuna results saved to: {excel_path}")

def plot_optuna_study(study, output_dir, model_name):
    fig1 = vis.plot_optimization_history(study)
    fig1_path = os.path.join(output_dir, model_name, "optimization_history.html")
    fig1.write_html(fig1_path)
    print(f"Optimization history plot saved to: {fig1_path}")

    fig2 = vis.plot_param_importances(study)
    fig2_path = os.path.join(output_dir, model_name, "parameter_importances.html")
    fig2.write_html(fig2_path)
    print(f"Parameter importances plot saved to: {fig2_path}")

    fig1.savefig(os.path.join(output_dir, model_name, "optimization_history.png"))
    fig2.savefig(os.path.join(output_dir, model_name, "parameter_importances.png"))
    print("Plots also saved as PNG files.")

if __name__ == "__main__":
    print("=== Starting Optuna search WITH heavy augmentation (Optimizing Validation Loss) ===")
    study_aug = run_optuna_with_augment_loss(
        model_name="lenet",
        n_trials=20,
        output_dir="optuna_aug_results"
    )
    plot_optuna_study(study_aug, "optuna_aug_results", "lenet")

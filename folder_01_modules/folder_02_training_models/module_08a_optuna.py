import optuna
import os
import torch
import optuna.visualization as vis
import pandas as pd
from module_06_train_model import train_model


def objective_no_augment_loss(trial, model_name, output_dir):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    gamma = trial.suggest_float("gamma", 0.6, 1.0)

    # Keine Augmentation
    augment_probability = 0.0
    augment_times = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trainieren des Modells und Rückgabe der Validierungsmetriken
    avg_val_loss, avg_val_acc, epoch_metrics = train_model(
        model_name=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=5,  # Kurze Tuning-Phase
        device=device,
        output_dir=output_dir,
        use_batchnorm = use_batchnorm,
        num_workers=0,
        persistent_workers=False,
        augment_probability=augment_probability,
        augment_times=augment_times,
        max_grad_cam_images=0,  # Grad-CAM deaktivieren
        step_size=3,
        gamma=gamma,
        n_folds=3,
        save_model=False,
        save_images=False,
        patience=None
    )

    avg_train_loss = epoch_metrics["train_losses"][-1]
    avg_train_acc = epoch_metrics["train_accuracies"][-1]

    trial.set_user_attr("validation_accuracy", avg_val_acc)
    trial.set_user_attr("train_loss", avg_train_loss)
    trial.set_user_attr("train_accuracy", avg_train_acc)
    return avg_val_loss


def run_optuna_no_augment_loss(model_name="lenet", n_trials=10, output_dir="optuna_noaug_results"):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=19))
    study.optimize(lambda trial: objective_no_augment_loss(trial, model_name, output_dir), n_trials=n_trials)
    save_optuna_results_to_excel(study, output_dir, model_name, heavy_augment=False)
    return study


def save_optuna_results_to_excel(study, output_dir, model_name, heavy_augment=False):
    results = []
    for trial in study.trials:
        trial_dict = {
            "Trial": trial.number,
            "Batch_size": trial.params.get("batch_size"),
            "Use_BatchNorm": trial.params.get("use_batchnorm"),
            "Learning_rate": trial.params.get("learning_rate"),
            "Gamma": trial.params.get("gamma"),
            "Val_loss": trial.value,
            "Val_accuracy": trial.user_attrs.get("validation_accuracy"),
            "Train_loss": trial.user_attrs.get("train_loss"),
            "Train_accuracy": trial.user_attrs.get("train_accuracy")
        }
        if heavy_augment:
            trial_dict["Augment_times"] = trial.params.get("augment_times")
        results.append(trial_dict)

    df = pd.DataFrame(results)
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    df.to_excel(os.path.join(model_dir, "optuna_results.xlsx"), index=False)
    print(f"Optuna results saved to: {os.path.join(model_dir, 'optuna_results.xlsx')}")


def plot_optuna_study(study, output_dir, model_name):
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    fig1 = vis.plot_optimization_history(study)
    fig1.write_html(os.path.join(model_dir, "optimization_history.html"))
    print(f"Optimization history plot saved to: {os.path.join(model_dir, 'optimization_history.html')}")

    fig2 = vis.plot_param_importances(study)
    fig2.write_html(os.path.join(model_dir, "parameter_importances.html"))
    print(f"Parameter importances plot saved to: {os.path.join(model_dir, 'parameter_importances.html')}")


if __name__ == "__main__":
    print("=== Starting Optuna search WITHOUT augmentation (Optimizing Validation Loss) ===")
    study_no_aug = run_optuna_no_augment_loss(
        model_name="lenet",
        n_trials=50,
        output_dir="optuna_noaug_results_samp_50"
    )
    plot_optuna_study(study_no_aug, "optuna_noaug_results_samp_50", "resnet10")

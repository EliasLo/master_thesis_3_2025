from module_06_train_model import train_model
from module_03_helper_functions import set_seed
import torch


if __name__ == "__main__":
    set_seed(19)

    model_name = "resnet10"
    learning_rate = 1e-4
    batch_size = 16
    epochs = 1
    num_workers = 4
    persistent_workers = False
    use_batchnorm = False


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        model_name=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        output_dir="results",
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        augment_times=1,
        augment_probability=0.0,
        max_grad_cam_images=80,
        step_size=2,
        gamma=0.80,
        use_batchnorm=use_batchnorm,
        patience=3
    )

#!/bin/sh
#SBATCH --time 60

. /opt/conda/etc/profile.d/conda.sh
conda activate Challenge2

hostname
python3 --version

cd /home/ai23m019/master_thesis_3_2025/folder_01_modules/folder_02_training_models/module_08a_optuna
python "$@"
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Constants
BASE_PATH = "../../folder_04_extracted_players/folder_01_training_data"
TRAINING_FOLDERS = ['data_03_players_kaggle']
COLORS = ['black', 'blue', 'green', 'orange', 'red', 'white', 'yellow']
TARGET_COUNT = 100 #150,200,250
RANDOM_STATE = 19
K_FOLDS = 3


def resample_group(group, target, random_state):
    """
    Resample a DataFrame group to the target number of samples.
    """
    current_count = len(group)
    if current_count > target:
        # Downsample without replacement
        return group.sample(n=target, replace=False, random_state=random_state)
    elif current_count < target:
        # Upsample with replacement
        return group.sample(n=target, replace=True, random_state=random_state)
    else:
        # No resampling needed
        return group


def create_dataframe_with_folds(
        base_path=BASE_PATH,
        training_folders=TRAINING_FOLDERS,
        colors=COLORS,
        target_count=TARGET_COUNT,
        random_state=RANDOM_STATE,
        k_folds=K_FOLDS
):
    """
    Erzeugt und verarbeitet das DataFrame sowie die Folds.

    Neu: Die HSV-Werte werden direkt aus den zugehörigen .txt-Dateien
    (z. B. yellow_hsv_info.txt) ausgelesen. Dabei wird nur für Bilder,
    die im Ordner tatsächlich vorhanden sind, der entsprechende HSV-Wert
    übernommen. Unbenutzte Einträge aus der .txt-Datei bleiben unberücksichtigt.
    """
    data = []

    # Durchlaufe alle Trainingsordner und Farben
    for training_folder in training_folders:
        training_folder_path = os.path.abspath(os.path.join(base_path, training_folder))
        if not os.path.exists(training_folder_path):
            print(f"Folder not found: {training_folder_path}")
            continue

        for color in colors:
            color_folder_path = os.path.join(training_folder_path, color)
            if not os.path.exists(color_folder_path):
                print(f"Color folder not found: {color_folder_path}")
                continue

            # Erstelle ein Mapping für die HSV-Werte aus der .txt-Datei
            hsv_mapping = {}
            hsv_txt_file = os.path.join(color_folder_path, f"{color}_hsv_info.txt")
            if os.path.exists(hsv_txt_file):
                with open(hsv_txt_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Erwartetes Format:
                        # "14_part5_sec125_frame612_player_1.jpg: H=27, S=190, V=110"
                        try:
                            filename_part, hsv_part = line.split(":", 1)
                            filename = filename_part.strip()
                            hsv_values = {}
                            for item in hsv_part.split(","):
                                key, value = item.strip().split("=")
                                hsv_values[key.lower()] = int(value)
                            if all(k in hsv_values for k in ['h', 's', 'v']):
                                hsv_mapping[filename] = hsv_values
                        except Exception as e:
                            print(f"Error parsing line in {hsv_txt_file}: {line} ({e})")
            else:
                print(f"HSV info file not found: {hsv_txt_file}")

            # Gehe alle Dateien im Farbordner durch
            for file_name in os.listdir(color_folder_path):
                # Überspringe die .txt-Datei
                if file_name.lower().endswith(".txt"):
                    continue
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(color_folder_path, file_name)
                    # Bildbeschreibung: Spiel-ID wird aus dem Dateinamen extrahiert
                    game_id = f"game_{file_name.split('_')[0]}"

                    # Hole die HSV-Werte, falls vorhanden
                    hsv_info = hsv_mapping.get(file_name, {})
                    h = hsv_info.get('h', None)
                    s = hsv_info.get('s', None)
                    v = hsv_info.get('v', None)

                    data.append({
                        'path': file_path,
                        'file_name': file_name,
                        'color': color,
                        'game': game_id,
                        'h': h,
                        's': s,
                        'v': v
                    })

    # Erstelle das initiale DataFrame
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("No valid data found!")

    # Kombinierter Schlüssel aus Farbe und Spiel-ID
    df['color_and_game'] = df['color'] + '_' + df['game']

    # Resample: Balanciere jede Gruppe auf target_count Bilder
    df_resampled = df.groupby('color_and_game').apply(
        lambda group: resample_group(group, target_count, random_state)
    ).reset_index(drop=True)

    # Erstelle Folds mit StratifiedKFold (Basierend auf der Farbe)
    unique_values = sorted(df_resampled["color_and_game"].unique())
    df_fold = pd.DataFrame({
        "color_and_game": unique_values,
        "color": [x.split('_')[0] for x in unique_values]
    })

    df_fold['color_group'] = df_fold['color'].astype("category").cat.codes
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    df_fold['fold'] = -1
    for fold, (_, test_idx) in enumerate(skf.split(df_fold, df_fold['color_group'])):
        df_fold.loc[test_idx, 'fold'] = fold

    # Füge die Fold-Zuordnung dem resampleten DataFrame hinzu
    df_resampled = pd.merge(df_resampled, df_fold[['color_and_game', 'fold']],
                            on='color_and_game', how='left')

    # Prüfung auf fehlende HSV-Werte
    missing_hsv = df_resampled[['h', 's', 'v']].isnull().any(axis=1).sum()
    if missing_hsv > 0:
        print(f"Warning: {missing_hsv} entries have missing HSV data.")
        missing_hsv_rows = df_resampled[df_resampled[['h', 's', 'v']].isnull().any(axis=1)]
        print("Rows with missing HSV data:")
        print(missing_hsv_rows[['file_name', 'color', 'game', 'h', 's', 'v']])

    return df_resampled


if __name__ == "__main__":
    df_train = create_dataframe_with_folds()
    print(f"DataFrame created with {len(df_train)} entries.")

    # Gruppiere nach 'color_and_game' und 'fold' und zähle die Einträge
    game_fold_counts = df_train.groupby(['color_and_game', 'fold']).size().reset_index(name='count')
    game_fold_counts = game_fold_counts.sort_values(by=['fold', 'color_and_game'])

    print("Count of each game in the folds (sorted by fold):")
    print(game_fold_counts)

    print("Columns in the final DataFrame:")
    print(df_train.columns)
    print(df_train.head())

import os
import cv2
import numpy as np
import pandas as pd

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
BASE_PATH = "../../folder_04_extracted_players_processed/folder_01_training_data"
TRAINING_FOLDERS = ['data_01_players_kaggle', 'data_03_players_supervisor']
COLORS = ['black', 'blue', 'green', 'orange', 'red', 'white', 'yellow']
HSV_EXCEL_FILE = "game_hsv.xlsx"  # The Excel file with columns 'color_and_game' and 'h_offset'


def gather_image_data(base_path=BASE_PATH, training_folders=TRAINING_FOLDERS, colors=COLORS):
    """
    Gathers image paths from the specified training_folders (each containing color subfolders).
    Returns a DataFrame with columns ['path', 'color', 'game', 'color_and_game'].

    Args:
        base_path (str): Base directory for the data.
        training_folders (list): List of subfolder names, e.g. ['data_01_players_kaggle'].
        colors (list): Color names, e.g. ['black', 'blue'].

    Returns:
        pd.DataFrame: Contains image path info and color_and_game for merging.
    """
    data = []

    for folder in training_folders:
        folder_path = os.path.join(base_path, folder)

        if not os.path.exists(folder_path):
            print(f"[DEBUG] Folder not found: {folder_path}")
            continue

        for color in colors:
            color_folder_path = os.path.join(folder_path, color)

            if not os.path.exists(color_folder_path):
                print(f"[DEBUG] Color folder not found: {color_folder_path}")
                continue

            for file_name in os.listdir(color_folder_path):
                file_lower = file_name.lower()
                if file_lower.endswith(('.png', '.jpg', '.jpeg')):
                    # Example filename: "65_part2_sec185_frame3168_player_0.jpg"
                    # We take everything up to the first '_' to determine the game ID:
                    game_part = file_name.split("_")[0]  # e.g. "65"
                    game_id = f"game_{game_part}"        # e.g. "game_65"

                    file_path = os.path.join(color_folder_path, file_name)
                    color_and_game = f"{color}_{game_id}"

                    data.append({
                        'path': file_path,
                        'color': color,
                        'game': game_id,
                        'color_and_game': color_and_game
                    })

    df = pd.DataFrame(data)
    return df


def shift_hue_inplace(image_path, hue_offset):
    """
    Loads an image from 'image_path', shifts its Hue channel by 'hue_offset',
    and overwrites the image at the same path.

    Args:
        image_path (str): Path to the image file.
        hue_offset (int): Hue offset in [0..179], shifting is done modulo 180.

    Returns:
        bool: True if the image was successfully written, otherwise False.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[DEBUG] Warning: Could not read image: {image_path}")
        return False

    # Convert BGR -> HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Split channels and convert to int
    h, s, v = cv2.split(img_hsv)
    h = h.astype(int)
    s = s.astype(int)
    v = v.astype(int)

    # Shift Hue channel (modulo 180)
    h = (h + hue_offset) % 180

    # Convert back to uint8
    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    # Merge channels and convert back to BGR
    shifted_hsv = cv2.merge([h, s, v])
    shifted_bgr = cv2.cvtColor(shifted_hsv, cv2.COLOR_HSV2BGR)

    # Overwrite the file at the same path
    success = cv2.imwrite(image_path, shifted_bgr)
    if not success:
        print(f"[DEBUG] Error writing file: {image_path}")
    return success


def apply_hue_shifts(df, path_col="path", offset_col="h_offset"):
    """
    Iterates over the rows of the DataFrame. For each row with a nonzero hue offset,
    applies 'shift_hue_inplace' to overwrite the image.

    Prints debug info about how many images get processed.

    Args:
        df (pd.DataFrame): Must contain columns path_col (image path) and offset_col (hue offset).
        path_col (str): Name of the column with the image path.
        offset_col (str): Name of the column with the hue offset.
    """
    total_rows = len(df)
    processed_count = 0

    # For debugging, let's see how many rows have offset != 0
    nonzero_df = df[df[offset_col] != 0]
    print(f"[DEBUG] Total rows in DataFrame: {total_rows}")
    print(f"[DEBUG] Rows with nonzero '{offset_col}': {len(nonzero_df)}")

    for idx, row in df.iterrows():
        offset = row[offset_col]
        # Skip offset == 0
        if offset == 0:
            continue

        image_path = row[path_col]
        print(f"[DEBUG] Shifting hue by {offset} for image: {image_path}")
        ok = shift_hue_inplace(image_path, offset)
        if ok:
            processed_count += 1
        else:
            print(f"[DEBUG] Hue shift failed for: {image_path}, offset={offset}")

    print(f"[DEBUG] apply_hue_shifts finished. Processed {processed_count} images.")


def main():
    # 1) Gather image data
    df = gather_image_data(BASE_PATH, TRAINING_FOLDERS, COLORS)
    if df.empty:
        print("[DEBUG] No images found in the specified folders.")
        return
    print(f"[DEBUG] Found {len(df)} images in the specified folders.")

    # 2) Load the Excel file with columns: color_and_game, h_offset
    hsv_file = os.path.join(BASE_PATH, HSV_EXCEL_FILE)
    if not os.path.exists(hsv_file):
        print(f"[DEBUG] HSV file not found: {hsv_file}")
        return

    df_hsv = pd.read_excel(hsv_file)
    if 'color_and_game' not in df_hsv.columns or 'h_offset' not in df_hsv.columns:
        print("[DEBUG] Error: The Excel file must have 'color_and_game' and 'h_offset' columns.")
        return

    print(f"[DEBUG] HSV DataFrame loaded, shape: {df_hsv.shape}")
    print("[DEBUG] Head of HSV DataFrame:")
    print(df_hsv.head(5))

    # 3) Merge the h_offset info into our main DataFrame
    df_merged = pd.merge(df, df_hsv[['color_and_game', 'h_offset']],
                         on='color_and_game', how='left')
    print(f"[DEBUG] After merge, shape: {df_merged.shape}")

    # Check for missing offsets
    missing_offsets = df_merged['h_offset'].isna().sum()
    if missing_offsets > 0:
        print(f"[DEBUG] Warning: {missing_offsets} images have no 'h_offset' information. They will be skipped.")

    # 4) Apply hue shifts only where h_offset != 0
    #    Drop rows with NaN in 'h_offset' to avoid errors
    df_to_shift = df_merged.dropna(subset=['h_offset'])
    apply_hue_shifts(df_to_shift, path_col="path", offset_col="h_offset")

    print("[DEBUG] Done. Images with nonzero h_offset were modified in-place.")


if __name__ == "__main__":
    main()

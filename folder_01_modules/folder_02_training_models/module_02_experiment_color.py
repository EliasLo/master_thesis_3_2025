import os
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from module_01_create_train_df import create_dataframe_with_folds

###############################################################################
# 1) HSV ranges (possibly adjusted)
###############################################################################
# Hue: 0..179, Saturation: 0..255, Value: 0..255
color_hsv_ranges = {
    "red": {
        "h_min": 0, "h_max": 3,
        "s_min": 110, "s_max": 255,
        "v_min": 150, "v_max": 255
    },
    "orange": {
        "h_min": 10, "h_max": 18,
        "s_min": 110, "s_max": 255,
        "v_min": 150, "v_max": 255
    },
    "yellow": {
        "h_min": 24, "h_max": 30,
        "s_min": 110, "s_max": 255,
        "v_min": 150, "v_max": 200
    },
    "green": {
        "h_min": 33, "h_max": 80,
        "s_min": 110, "s_max": 255,
        "v_min": 150, "v_max": 255
    },
    "blue": {
        "h_min": 90, "h_max": 140,
        "s_min": 110, "s_max": 255,
        "v_min": 150, "v_max": 255
    },
    "white": {
        "h_min": 0, "h_max": 179,
        "s_min": 0, "s_max": 30,
        "v_min": 210, "v_max": 255
    },
    "black": {
        "h_min": 0, "h_max": 179,
        "s_min": 0, "s_max": 60,
        "v_min": 0, "v_max": 40
    }
}

###############################################################################
# Individual shift caps for each color
###############################################################################
color_shift_caps = {
    "red": {"h": 10, "s": 100, "v": 100},
    "orange": {"h": 6, "s": 100, "v": 100},
    "yellow": {"h": 6, "s": 100, "v": 100},
    "green": {"h": 12, "s": 100, "v": 100},
    "blue": {"h": 12, "s": 100, "v": 100},
    "white": {"h": 20, "s": 30, "v": 100},
    "black": {"h": 15, "s": 30, "v": 100}
}

###############################################################################
# 2) Create output directories
###############################################################################
def create_output_directories(base_dir, colors):
    """
    Creates directories for each color and returns a dictionary mapping each color to its directory path.
    """
    dir_structure = {}
    for color in colors:
        color_dir = os.path.join(base_dir, color)
        os.makedirs(color_dir, exist_ok=True)
        dir_structure[color] = color_dir
    return dir_structure


def generate_output_directory(base_output_dir="augmentation/augmentation_experiment/colors"):
    """
    Ensures the base output directory exists.
    """
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    return base_output_dir

###############################################################################
# 3) Load and filter training data
###############################################################################
def load_training_data(colors):
    """
    Loads and filters training data for the specified colors.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only the specified colors.
    """
    df = create_dataframe_with_folds()

    print("[INFO] Example entries of the DataFrame:")
    print(df.head())

    filtered_df = df[df["color"].isin(colors)]
    if filtered_df.empty:
        raise ValueError("No data found for the selected colors.")

    print("\n[INFO] Columns of the filtered DataFrame:", filtered_df.columns.tolist())
    print("[INFO] Example of the filtered DataFrame:\n", filtered_df.head())

    return filtered_df

###############################################################################
# 4) HSV Shift Function (with individual shift caps per color and adjusted allowed shift if out-of-range)
###############################################################################
def get_hsv_shifts(orig_h, orig_s, orig_v, color):
    # Determine the underlying HSV ranges
    if color == "red":
        # Special case "red": distinguish between 0-10 and 170-179
        if 170 <= orig_h <= 179:
            h_min, h_max = 170, 179
        else:
            h_min, h_max = 0, 3
    else:
        rng = color_hsv_ranges[color]
        h_min = rng["h_min"]
        h_max = rng["h_max"]

    s_min = color_hsv_ranges[color]["s_min"]
    s_max = color_hsv_ranges[color]["s_max"]
    v_min = color_hsv_ranges[color]["v_min"]
    v_max = color_hsv_ranges[color]["v_max"]

    # Get the specific shift caps for the current color
    caps = color_shift_caps.get(color, {"h": 17, "s": 30, "v": 30})
    SHIFT_CAP_H = caps["h"]
    SHIFT_CAP_S = caps["s"]
    SHIFT_CAP_V = caps["v"]

    # --- For Hue ---
    if orig_h > h_max:
        # If original is above the allowed maximum, force a shift to bring it to h_max.
        delta_h = h_max - orig_h
    elif orig_h < h_min:
        # If original is below the allowed minimum, force a shift to bring it to h_min.
        delta_h = h_min - orig_h
    else:
        # Original value is within range: allow a random shift within the allowed limits.
        allowed_min_h = h_min - orig_h
        allowed_max_h = h_max - orig_h
        allowed_min_h = max(allowed_min_h, -SHIFT_CAP_H)
        allowed_max_h = min(allowed_max_h, SHIFT_CAP_H)
        if allowed_min_h > allowed_max_h:
            allowed_min_h, allowed_max_h = 0, 0
        delta_h = random.randint(allowed_min_h, allowed_max_h)

    # --- For Saturation ---
    if orig_s > s_max:
        delta_s = s_max - orig_s
    elif orig_s < s_min:
        delta_s = s_min - orig_s
    else:
        allowed_min_s = s_min - orig_s
        allowed_max_s = s_max - orig_s
        allowed_min_s = max(allowed_min_s, -SHIFT_CAP_S)
        allowed_max_s = min(allowed_max_s, SHIFT_CAP_S)
        if allowed_min_s > allowed_max_s:
            allowed_min_s, allowed_max_s = 0, 0
        delta_s = random.randint(allowed_min_s, allowed_max_s)

    # --- For Value ---
    if orig_v > v_max:
        delta_v = v_max - orig_v
    elif orig_v < v_min:
        delta_v = v_min - orig_v
    else:
        allowed_min_v = v_min - orig_v
        allowed_max_v = v_max - orig_v
        allowed_min_v = max(allowed_min_v, -SHIFT_CAP_V)
        allowed_max_v = min(allowed_max_v, SHIFT_CAP_V)
        if allowed_min_v > allowed_max_v:
            allowed_min_v, allowed_max_v = 0, 0
        delta_v = random.randint(allowed_min_v, allowed_max_v)

    return delta_h, delta_s, delta_v

###############################################################################
# 5) Convert image to HSV, apply shift, convert back to RGB
###############################################################################
def shift_image_hsv(image, delta_h, delta_s, delta_v):
    """
    Applies the HSV shifts (delta_h, delta_s, delta_v) to the entire image.
    The image is converted to HSV, the shifts are applied, values are clamped,
    and then the image is converted back to RGB.
    """
    img_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)

    # Convert channels to int32 to handle negative values.
    h = h.astype(np.int32)
    s = s.astype(np.int32)
    v = v.astype(np.int32)

    # Apply the shifts.
    h += delta_h
    s += delta_s
    v += delta_v

    # Clamp the channels to their valid ranges.
    h = np.clip(h, 0, 179)
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)

    # Convert back to uint8.
    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    new_hsv = cv2.merge([h, s, v])
    aug_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(aug_rgb)

###############################################################################
# 6) Perform augmentation and save images
###############################################################################
def augment_and_save(dataframe, colors, output_dir, max_images_per_game=5):
    """
    Iterates over the DataFrame and saves augmented images.
    For each game and color, up to `max_images_per_game` images are selected for augmentation.
    """
    dir_structure = create_output_directories(output_dir, colors)
    original_dir = os.path.join(output_dir, "originals")
    os.makedirs(original_dir, exist_ok=True)
    total_augmented = 0

    for color in colors:
        cdf = dataframe[dataframe["color"] == color]
        games = cdf["game"].unique()
        print(f"\n[INFO] Processing color: {color}, Number of games: {len(games)}")

        for game in games:
            game_df = cdf[cdf["game"] == game]

            # Select up to max_images_per_game images for this game and color.
            selected_rows = game_df.sample(n=min(len(game_df), max_images_per_game), random_state=19)

            for _, row in selected_rows.iterrows():
                img_path = row["path"]
                orig_h = row["h"]
                orig_s = row["s"]
                orig_v = row["v"]

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Error opening {img_path}: {e}")
                    continue

                # Save the original image.
                original_filename = os.path.splitext(os.path.basename(img_path))[0]
                original_save_filename = f"{original_filename}_original_H({orig_h})_S({orig_s})_V({orig_v}).jpg"
                original_save_path = os.path.join(original_dir, original_save_filename)
                try:
                    image.save(original_save_path)
                except Exception as e:
                    print(f"Error saving original image {original_save_path}: {e}")
                    continue

                # 1) Determine the HSV shifts.
                delta_h, delta_s, delta_v = get_hsv_shifts(orig_h, orig_s, orig_v, color)

                # 2) Apply the HSV shifts to the image.
                aug_image = shift_image_hsv(image, delta_h, delta_s, delta_v)

                # For naming, clip the original HSV values to allowed ranges.
                if color == "red":
                    if 170 <= orig_h <= 179:
                        h_min, h_max = 170, 179
                    else:
                        h_min, h_max = 0, 10
                else:
                    h_min = color_hsv_ranges[color]["h_min"]
                    h_max = color_hsv_ranges[color]["h_max"]

                s_min = color_hsv_ranges[color]["s_min"]
                s_max = color_hsv_ranges[color]["s_max"]
                v_min = color_hsv_ranges[color]["v_min"]
                v_max = color_hsv_ranges[color]["v_max"]

                clipped_h = int(np.clip(orig_h, h_min, h_max))
                clipped_s = int(np.clip(orig_s, s_min, s_max))
                clipped_v = int(np.clip(orig_v, v_min, v_max))

                # Calculate the final HSV values using the clipped originals.
                final_h = max(0, min(179, clipped_h + delta_h))
                final_s = max(0, min(255, clipped_s + delta_s))
                final_v = max(0, min(255, clipped_v + delta_v))

                # 3) Save the augmented image.
                save_filename = f"{original_filename}_H({final_h})_S({final_s})_V({final_v}).jpg"
                save_path = os.path.join(dir_structure[color], save_filename)

                try:
                    aug_image.save(save_path)
                    total_augmented += 1
                except Exception as e:
                    print(f"Error saving {save_path}: {e}")

    print(f"\n[INFO] Augmentation complete! {total_augmented} images generated.")

###############################################################################
# 7) MAIN
###############################################################################
if __name__ == "__main__":
    # List of colors to process.
    colors_to_use = ["white"]

    # 1) Load training data.
    try:
        df = load_training_data(colors_to_use)
        print(f"[INFO] Training data loaded with {len(df)} entries.")
    except ValueError as ve:
        print(f"Error loading training data: {ve}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)

    # 2) Create the output directory.
    output_base = generate_output_directory("augmentation/augmentation_experiment/colors")

    # 3) Perform augmentation and save images.
    try:
        augment_and_save(
            dataframe=df,
            colors=colors_to_use,
            output_dir=output_base,
            max_images_per_game=150
        )
    except Exception as e:
        print(f"Error during augmentation: {e}")

    # Optional: Print DataFrame columns and some unique info for debugging
    color = "red"
    cdf = df[df["color"] == color]
    print("[DEBUG] Columns:", cdf.columns)
    games = cdf["game"].unique()
    print("[DEBUG] Unique games for color 'red':", games)

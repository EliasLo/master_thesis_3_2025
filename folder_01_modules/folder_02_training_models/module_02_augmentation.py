import os
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFilter
import torch
import torchvision.transforms as transforms
from datetime import datetime

# Importiere die Funktion zum Erstellen des DataFrames
from module_01_create_train_df import create_dataframe_with_folds

###############################################################################
# 1) HSV color ranges (OpenCV-Konvention)
# Diese Bereiche gelten für den dominanten Farbton (z. B. T-Shirt).
# Hue: 0..179, Saturation: 0..255, Value: 0..255
###############################################################################
color_hsv_ranges = {
    "red": {
        "h_min": 0, "h_max": 3,
        "s_min": 110, "s_max": 255,
        "v_min": 150, "v_max": 255
    },
    "orange": {
        "h_min": 10, "h_max": 18,
        "s_min": 110, "s_max": 200,
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
        "s_min": 0, "s_max": 20,
        "v_min": 220, "v_max": 255
    },
    "black": {
        "h_min": 0, "h_max": 179,
        "s_min": 0, "s_max": 60,
        "v_min": 0, "v_max": 40
    }
}

# Mapping von Farben zu Label-IDs (wie benötigt)
color_mapping = {
    "red": 0,
    "black": 1,
    "yellow": 2,
    "orange": 3,
    "green": 4,
    "blue": 5,
    "white": 6
}

###############################################################################
# 2) Default shift caps per color (experimentelle Werte)
###############################################################################
def get_default_shift_caps(color):
    shift_caps = {
        "red": {"h": 15, "s": 100, "v": 100},
        "orange": {"h": 10, "s": 30, "v": 30},
        "yellow": {"h": 10, "s": 100, "v": 100},
        "green": {"h": 12, "s": 100, "v": 100},
        "blue": {"h": 15, "s": 100, "v": 100},
        "white": {"h": 20, "s": 30, "v": 30},
        "black": {"h": 20, "s": 30, "v": 30}
    }
    return shift_caps.get(color, {"h": 17, "s": 30, "v": 30})

###############################################################################
# 3) Helper: HSV shift calculation (mit korrektem Clipping und reduzierter
#    effektiver Verschiebung, wenn der Originalwert außerhalb liegt)
###############################################################################
def get_hsv_shifts(orig_h, orig_s, orig_v, color):
    rng = color_hsv_ranges[color]

    # Spezielle Behandlung für red (zwei mögliche Hue-Bereiche)
    if color == "red":
        if 162 <= orig_h <= 179:
            h_min, h_max = 170, 179
        else:
            h_min, h_max = 0, 3
    else:
        h_min = rng["h_min"]
        h_max = rng["h_max"]

    s_min, s_max = rng["s_min"], rng["s_max"]
    v_min, v_max = rng["v_min"], rng["v_max"]

    caps = get_default_shift_caps(color)
    SHIFT_CAP_H = caps["h"]
    SHIFT_CAP_S = caps["s"]
    SHIFT_CAP_V = caps["v"]

    # Für Hue: Berechne den erlaubten Bereich relativ zum Originalwert.
    allowed_min_h = max(h_min - orig_h, -SHIFT_CAP_H)
    allowed_max_h = min(h_max - orig_h, SHIFT_CAP_H)
    if allowed_min_h > allowed_max_h:
        delta_h = 0
    else:
        delta_h = random.randint(allowed_min_h, allowed_max_h)

    # Für Saturation:
    allowed_min_s = max(s_min - orig_s, -SHIFT_CAP_S)
    allowed_max_s = min(s_max - orig_s, SHIFT_CAP_S)
    if allowed_min_s > allowed_max_s:
        delta_s = 0
    else:
        delta_s = random.randint(allowed_min_s, allowed_max_s)

    # Für Value:
    allowed_min_v = max(v_min - orig_v, -SHIFT_CAP_V)
    allowed_max_v = min(v_max - orig_v, SHIFT_CAP_V)
    if allowed_min_v > allowed_max_v:
        delta_v = 0
    else:
        delta_v = random.randint(allowed_min_v, allowed_max_v)

    return delta_h, delta_s, delta_v

###############################################################################
# 4) Function to shift the image in HSV space
###############################################################################
def shift_image_hsv(image, delta_h, delta_s, delta_v):
    """
    Applies the HSV shifts (delta_h, delta_s, delta_v) to the entire image.
    The image is converted to HSV, the shifts are applied (with clamping), and then
    the image is converted back to RGB.
    """
    image_np = np.array(image, dtype=np.uint8)
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.int32)
    h, s, v = cv2.split(image_hsv)

    h += delta_h
    s += delta_s
    v += delta_v

    h = np.clip(h, 0, 179)
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)

    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    new_hsv = cv2.merge([h, s, v])
    aug_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(aug_rgb)

###############################################################################
# 5) Weather effects (optional)
###############################################################################
def apply_single_weather_effect(image, effect_type):
    if effect_type == "fog":
        fog_intensity = random.uniform(0.2, 0.5)
        image = transforms.functional.adjust_contrast(image, 1 - fog_intensity)
        image = transforms.functional.adjust_brightness(image, 1 + 0.65 * fog_intensity)
    elif effect_type == "overexposure":
        overlay = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.blend(image, overlay, alpha=random.uniform(0.4, 0.6))
    elif effect_type == "snow":
        snow_intensity = random.uniform(0.05, 0.2)
        try:
            t = transforms.ToTensor()(image)
            t = t + torch.randn_like(t) * snow_intensity
            image = transforms.ToPILImage()(torch.clamp(t, 0, 1))
        except Exception as e:
            print(f"Error applying snow effect: {e}")
    elif effect_type == "rain":
        alpha_val = random.uniform(0.1, 0.2)
        drops_val = random.randint(30, 100)
        width, height = image.size
        rain_layer = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(rain_layer)
        for _ in range(drops_val):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = x1 + random.randint(-2, 2)
            y2 = y1 + random.randint(20, 60)
            gray_value = random.randint(225, 255)
            draw.line([(x1, y1), (x2, y2)], fill=(gray_value, gray_value, gray_value), width=random.randint(1, 2))
        rain_layer = rain_layer.filter(ImageFilter.GaussianBlur(0.3))
        image = Image.blend(image, rain_layer, alpha=alpha_val)
    elif effect_type == "shadow":
        shadow_val = random.uniform(0.65, 0.85)
        image = transforms.functional.adjust_brightness(image, shadow_val)
    return image

###############################################################################
# 6) Custom Transform: HsvShift + Weather based on (orig_h, orig_s, orig_v)
###############################################################################
class HsvShiftAndWeatherTransform:
    """
    1) Uses get_hsv_shifts(...) to find a random (delta_h, delta_s, delta_v)
       within the color's valid range, based on (orig_h, orig_s, orig_v).
    2) Applies shift_image_hsv(...).
    3) With probability 'weather_probability', applies exactly ONE weather effect.
    """
    def __init__(self, color, orig_h, orig_s, orig_v, weather_probability=0.50):
        self.color = color
        self.orig_h = orig_h
        self.orig_s = orig_s
        self.orig_v = orig_v
        self.weather_probability = weather_probability
        self.weather_effects = ["fog", "overexposure", "snow", "rain", "shadow"]

    def __call__(self, img):
        delta_h, delta_s, delta_v = get_hsv_shifts(self.orig_h, self.orig_s, self.orig_v, self.color)
        img = shift_image_hsv(img, delta_h, delta_s, delta_v)
        if random.random() < self.weather_probability:
            effect = random.choice(self.weather_effects)
            img = apply_single_weather_effect(img, effect)
        return img

###############################################################################
# 7) Additional Transform: RemoveBottomTransform (optional)
###############################################################################
class RemoveBottomTransform:
    """
    Crops the bottom part of the image by a ratio, with a given probability.
    """
    def __init__(self, crop_ratio=0.40, probability=0.60):
        self.crop_ratio = crop_ratio
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            w, h = img.size
            new_h = int(h * (1 - self.crop_ratio))
            return img.crop((0, 0, w, new_h))
        return img

###############################################################################
# 8) Main transform pipeline for on-the-fly usage
###############################################################################
def get_transform_with_hsv(color, orig_h, orig_s, orig_v, weather_probability=0.5, train=True):
    transform_list = []
    if train:
        transform_list.append(RemoveBottomTransform(crop_ratio=0.40, probability=0.6))
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation(degrees=(-90, 90)))
        transform_list.append(transforms.Resize((224, 224))) #224x224
    transform_list.append(HsvShiftAndWeatherTransform(
        color=color,
        orig_h=orig_h,
        orig_s=orig_s,
        orig_v=orig_v,
        weather_probability=weather_probability
    ))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

###############################################################################
# 9) Offline augmentation example (with naming adapted to clipped HSV values)
###############################################################################
def generate_output_directory(base="augmentation/augmentation_training"):
    if not os.path.exists(base):
        os.makedirs(base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, f"aug_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def unnormalize(tensor, mean, std):
    out = tensor.clone()
    for t, m, s in zip(out, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(out, 0, 1)

def offline_augment_and_save(df, output_dir, sample_limit=3, augment_count=2, weather_probability=0.5):
    """
    For each color, take up to 'sample_limit' images,
    then create 'augment_count' augmented versions using our pipeline.
    Saves results to disk for visual inspection.
    The DataFrame (df) must contain columns: ["path", "color", "h", "s", "v", "label_id"]
    """
    for color, lbl_id in color_mapping.items():
        color_subdir = os.path.join(output_dir, f"{color}_augmented")
        os.makedirs(color_subdir, exist_ok=True)

        cdf = df[df["label_id"] == lbl_id]
        if len(cdf) == 0:
            continue

        selected_rows = cdf.sample(n=min(len(cdf), sample_limit), random_state=19)

        for _, row in selected_rows.iterrows():
            img_path = row["path"]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            color_val = row["color"]
            orig_h = row["h"]
            orig_s = row["s"]
            orig_v = row["v"]

            for i in range(augment_count):
                # Berechne die delta shifts (für Namensgebung)
                delta_h, delta_s, delta_v = get_hsv_shifts(orig_h, orig_s, orig_v, color_val)

                # Für Namensgebung: Clippe die Originalwerte auf die erlaubten Bereiche.
                if color_val == "red":
                    if 170 <= orig_h <= 179:
                        h_min, h_max = 170, 179
                    else:
                        h_min, h_max = 0, 10
                else:
                    h_min = color_hsv_ranges[color_val]["h_min"]
                    h_max = color_hsv_ranges[color_val]["h_max"]
                s_min = color_hsv_ranges[color_val]["s_min"]
                s_max = color_hsv_ranges[color_val]["s_max"]
                v_min = color_hsv_ranges[color_val]["v_min"]
                v_max = color_hsv_ranges[color_val]["v_max"]

                clipped_h = int(np.clip(orig_h, h_min, h_max))
                clipped_s = int(np.clip(orig_s, s_min, s_max))
                clipped_v = int(np.clip(orig_v, v_min, v_max))

                final_h = max(0, min(179, clipped_h + delta_h))
                final_s = max(0, min(255, clipped_s + delta_s))
                final_v = max(0, min(255, clipped_v + delta_v))

                transform = get_transform_with_hsv(
                    color=color_val,
                    orig_h=orig_h,
                    orig_s=orig_s,
                    orig_v=orig_v,
                    weather_probability=weather_probability,
                    train=True
                )
                aug_tensor = transform(image)
                aug_unorm = unnormalize(aug_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                aug_pil = transforms.ToPILImage()(aug_unorm)

                save_name = f"{base_name}_H({final_h})_S({final_s})_V({final_v})_aug_{i + 1}.jpg"
                save_path = os.path.join(color_subdir, save_name)
                aug_pil.save(save_path)

    print(f"\n[INFO] Offline augmentation complete!")

###############################################################################
# 10) Script backup (optional)
###############################################################################
def save_script_backup(output_dir, script_name=__file__):
    backup_path = os.path.join(output_dir, "script_backup.txt")
    with open(backup_path, "w") as backup_file:
        with open(script_name, "r") as script_file:
            backup_file.write(script_file.read())
    print(f"Script backup saved to: {backup_path}")

###############################################################################
# 11) MAIN
###############################################################################
if __name__ == "__main__":
    out_dir = generate_output_directory()
    print(f"Offline augmentation test -> saving results to: {out_dir}")

    df = create_dataframe_with_folds()
    df["label_id"] = df["color"].map(color_mapping)

    # Wir erwarten, dass df Spalten "h", "s", "v" enthält, damit get_hsv_shifts korrekt arbeitet.
    offline_augment_and_save(
        df=df,
        output_dir=out_dir,
        sample_limit=20,
        augment_count=10,
        weather_probability=0.5
    )

    try:
        save_script_backup(out_dir)
    except Exception as e:
        print(f"Backup error: {e}")

    print("Offline test complete.")

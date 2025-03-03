import os
import random
from datetime import datetime
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as transforms
import pandas as pd
import torch

# Falls du diese Funktion hast:
from module_01_create_train_df import create_dataframe_with_folds

###############################################################################
# 1) Wichtige Farbinformationen und Mappings (sofern benötigt)
###############################################################################
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
# 2) Helper-Funktion für die Wettereffekte (an den ersten Modul angepasst)
###############################################################################
def apply_single_weather_effect_experiment(
        image,
        effect_type,
        param_1=None,
        param_2=None
):
    """
    Repliziert das Verhalten aus dem ersten Modul (module02_augmentation.py),
    erlaubt aber systematische Parametersteuerung (param_1, param_2) für das Experiment.

    Args:
        image (PIL.Image): Eingang (RGB).
        effect_type (str): ["fog", "overexposure", "snow", "rain", "shadow"].
        param_1 (float): Haupt-Intensität, z.B. fog_intensity, alpha usw.
        param_2 (float, optional): Zweiter Parameter (z.B. number_of_drops für Regen).

    Returns:
        PIL.Image: Bild mit dem entsprechenden Wettereffekt (RGB).
    """
    if not isinstance(image, Image.Image):
        image = transforms.ToPILImage()(image)

    # --- Fog ---
    # Aus dem ersten Modul:
    # fog_intensity = random.uniform(0.2, 0.5)
    # contrast = 1 - fog_intensity, brightness = 1 + 0.65 * fog_intensity
    if effect_type == "fog" and param_1 is not None:
        fog_intensity = param_1
        image = transforms.functional.adjust_contrast(image, 1 - fog_intensity)
        image = transforms.functional.adjust_brightness(image, 1 + 0.65 * fog_intensity)

    # --- Overexposure ---
    # Aus dem ersten Modul:
    # overlay = weißes Bild; alpha = random.uniform(0.2, 0.4)
    elif effect_type == "overexposure" and param_1 is not None:
        overexposure_alpha = param_1
        overlay = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.blend(image, overlay, alpha=overexposure_alpha)

    # --- Snow ---
    # Aus dem ersten Modul:
    # snow_intensity = random.uniform(0.05, 0.2)
    # Add gauss-Rauschen t = t + torch.randn_like(t) * snow_intensity
    elif effect_type == "snow" and param_1 is not None:
        snow_intensity = param_1
        try:
            t = transforms.ToTensor()(image)
            t = t + torch.randn_like(t) * snow_intensity
            image = transforms.ToPILImage()(torch.clamp(t, 0, 1))
        except Exception as e:
            print(f"Error applying snow effect: {e}")

    # --- Rain ---
    # Aus dem ersten Modul:
    # alpha_val = random.uniform(0.1, 0.2)
    # drops_val = random.randint(30, 100)
    elif effect_type == "rain" and (param_1 is not None) and (param_2 is not None):
        rain_alpha = param_1
        rain_num_drops = int(param_2)
        width, height = image.size

        rain_layer = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(rain_layer)

        for _ in range(rain_num_drops):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = x1 + random.randint(-2, 2)
            y2 = y1 + random.randint(20, 60)
            gray_value = random.randint(225, 255)
            draw.line([(x1, y1), (x2, y2)], fill=(gray_value, gray_value, gray_value), width=random.randint(1, 2))

        rain_layer = rain_layer.filter(ImageFilter.GaussianBlur(0.20))
        image = Image.blend(image, rain_layer, alpha=rain_alpha)

    # --- Shadow ---
    # Aus dem ersten Modul:
    # shadow_val = random.uniform(0.65, 0.9)
    elif effect_type == "shadow" and param_1 is not None:
        shadow_val = param_1
        image = transforms.functional.adjust_brightness(image, shadow_val)

    return image


###############################################################################
# 3) Hilfsfunktion zum Erstellen einer sauberen Ordnerstruktur
###############################################################################
def create_output_directories(base_dir, colors, weather_types):
    """
    Erstellt Unterordner nach dem Muster:
        base_dir/<weather>/<color>/
    und gibt ein Dictionary zurück, damit wir Pfade sauber ansprechen können.
    """
    dir_structure = {}
    for weather in weather_types:
        weather_dir = os.path.join(base_dir, weather)
        os.makedirs(weather_dir, exist_ok=True)
        dir_structure[weather] = {}
        for color in colors:
            color_dir = os.path.join(weather_dir, color)
            os.makedirs(color_dir, exist_ok=True)
            dir_structure[weather][color] = color_dir
    return dir_structure


###############################################################################
# 4) Parameterbereiche definieren (an die Werte des ersten Moduls angelehnt)
###############################################################################
def generate_parameter_values(start, end, step):
    """
    Erzeugt Werte von start bis end (inklusiv) in Schrittweite step.
    Rundet auf 3 Nachkommastellen, um Gleitkommafehler zu minimieren.
    """
    values = []
    current = start
    while current <= end + 1e-9:  # kleine Toleranz
        values.append(round(current, 3))
        current += step
    return values


# Fog: random.uniform(0.2, 0.5) => Sweep in diesem Bereich
FOG_VALUES = generate_parameter_values(0.2, 0.5, 0.1)

# Overexposure: random.uniform(0.2, 0.4) => alpha-Sweep
OVEREXPOSURE_ALPHA_VALUES = generate_parameter_values(0.2, 0.8, 0.05)

# Snow: random.uniform(0.05, 0.2)
SNOW_VALUES = generate_parameter_values(0.05, 0.2, 0.05)

# Rain: alpha random.uniform(0.1, 0.2) + drops random.randint(30, 100)
RAIN_ALPHA_VALUES = generate_parameter_values(0.1, 0.2, 0.05)
RAIN_DROPS_VALUES = [50, 100]  # Direkte Werte

# Shadow: random.uniform(0.65, 0.9)
SHADOW_VALUES = generate_parameter_values(0.65, 0.85, 0.05)


###############################################################################
# 5) Kernfunktion für das Experiment
###############################################################################
def experiment_weather_by_parameter(
        dataframe,
        colors,
        weather_types,
        output_dir,
        max_images_per_game=5
):
    """
    1) Erzeugt Ordnerstruktur unter 'output_dir/<weather>/<color>'.
    2) Durchläuft Bilder pro Farbe/Spiel.
    3) Speichert Originalbild + systematische Variation (Param-Sweep).
    """

    # Ordnerstruktur anlegen
    dir_structure = create_output_directories(output_dir, colors, weather_types)

    for weather in weather_types:
        for color in colors:
            # DataFrame nach Farbe filtern
            color_df = dataframe[dataframe["color"] == color]
            games = color_df["game"].unique()

            for game in games:
                game_df = color_df[color_df["game"] == game]
                # Auf max_images_per_game begrenzen
                selected_rows = game_df.sample(n=min(len(game_df), max_images_per_game), random_state=19)

                for _, row in selected_rows.iterrows():
                    img_path = row["path"]
                    try:
                        image = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        print(f"Fehler beim Öffnen {img_path}: {e}")
                        continue

                    # Original-Bild immer nur einmal speichern (falls noch nicht vorhanden)
                    original_save_path = os.path.join(
                        dir_structure[weather][color],
                        f"{os.path.basename(img_path).split('.')[0]}_original.jpg"
                    )
                    if not os.path.exists(original_save_path):
                        image.save(original_save_path)

                    # Danach systematische Variation je nach Wetter:
                    if weather == "fog":
                        for fog_val in FOG_VALUES:
                            out_img = apply_single_weather_effect_experiment(
                                image, "fog", param_1=fog_val
                            )
                            save_path = os.path.join(
                                dir_structure[weather][color],
                                f"{os.path.basename(img_path).split('.')[0]}_{weather}_{fog_val}.jpg"
                            )
                            out_img.save(save_path)

                    elif weather == "overexposure":
                        for alpha_val in OVEREXPOSURE_ALPHA_VALUES:
                            out_img = apply_single_weather_effect_experiment(
                                image, "overexposure", param_1=alpha_val
                            )
                            save_path = os.path.join(
                                dir_structure[weather][color],
                                f"{os.path.basename(img_path).split('.')[0]}_{weather}_alpha{alpha_val}.jpg"
                            )
                            out_img.save(save_path)

                    elif weather == "snow":
                        for snow_val in SNOW_VALUES:
                            out_img = apply_single_weather_effect_experiment(
                                image, "snow", param_1=snow_val
                            )
                            save_path = os.path.join(
                                dir_structure[weather][color],
                                f"{os.path.basename(img_path).split('.')[0]}_{weather}_{snow_val}.jpg"
                            )
                            out_img.save(save_path)

                    elif weather == "rain":
                        for alpha_val in RAIN_ALPHA_VALUES:
                            for drops_val in RAIN_DROPS_VALUES:
                                out_img = apply_single_weather_effect_experiment(
                                    image, "rain", param_1=alpha_val, param_2=drops_val
                                )
                                save_path = os.path.join(
                                    dir_structure[weather][color],
                                    f"{os.path.basename(img_path).split('.')[0]}_{weather}_alpha{alpha_val}_drops{drops_val}.jpg"
                                )
                                out_img.save(save_path)

                    elif weather == "shadow":
                        for shadow_val in SHADOW_VALUES:
                            out_img = apply_single_weather_effect_experiment(
                                image, "shadow", param_1=shadow_val
                            )
                            save_path = os.path.join(
                                dir_structure[weather][color],
                                f"{os.path.basename(img_path).split('.')[0]}_{weather}_{shadow_val}.jpg"
                            )
                            out_img.save(save_path)


###############################################################################
# 6) MAIN
###############################################################################
if __name__ == "__main__":
    # Farben und Wettereffekte definieren (wie in deinem ersten Modul)
    colors = ["yellow", "red", "blue", "green", "orange", "white", "black"]
    weather_types = ["fog", "overexposure", "snow", "rain", "shadow"]

    # Basis-Ausgabeordner (frei anpassbar)
    output_dir = "augmentation/augmentation_experiment/weather_effects"
    os.makedirs(output_dir, exist_ok=True)

    # DataFrame erstellen (oder laden)
    df = create_dataframe_with_folds()
    df["label_id"] = df["color"].map(color_mapping)

    # Beispiel: experimentell nur bestimmte Farben => hier auskommentiert
    # df = df[df["color"].isin(colors)]

    # Experiment starten
    experiment_weather_by_parameter(
        dataframe=df,
        colors=colors,
        weather_types=weather_types,
        output_dir=output_dir,
        max_images_per_game=5  # Anzahl Bilder pro Spiel
    )

    print("Weather-Experiment abgeschlossen. Bilder liegen in:", output_dir)

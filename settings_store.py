import json
import os

SETTINGS_FILE = "app_settings.json"

DEFAULTS = {
    "confidence": 0.55,
    "hold_frames": 20,
    "buffer_size": 10,
    "camera_index": 0,
    "sound_volume": 70,
    "sound_muted": False,
    "suggestion_count": 4,
    "autostart_camers": True,
    "start_fullscreen": True,
    "language": "en",
    "model_path": "models/az_model.pkl"
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = DEFAULTS.copy()
            merged.update(data)
            return merged
        except (json.JSONDecodeError, IOError):
            return DEFAULTS.copy()
    return DEFAULTS.copy()

def save_settings(settings_dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=2, ensure_ascii=False)
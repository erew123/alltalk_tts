import os
import json
import sys
import importlib

# Hard-coded paths
this_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
config_file = os.path.join(this_folder, "confignew.json")
gradio_pages_folder = os.path.join(this_folder, "system", "gradio_pages", "themes")
theme_list_file = os.path.join(gradio_pages_folder, "theme_list.json")
themes_file_path = os.path.join(gradio_pages_folder, "themes.py")

# Add gradio_pages folder to the system path
sys.path.append(gradio_pages_folder)

def get_class(filename):
    with open(filename, "r", encoding="utf8") as file:
        for line in file:
            if "class " in line:
                return line.split("class ")[1].split(":")[0].split("(")[0].strip()
    return None


def get_list():
    themes_from_files = [
        os.path.splitext(name)[0]
        for name in os.listdir(gradio_pages_folder)
        if name.endswith(".py")
    ]
    try:
        with open(theme_list_file, "r", encoding="utf8") as json_file:
            themes_from_url = [item["id"] for item in json.load(json_file)]
    except FileNotFoundError:
        themes_from_url = []

    combined_themes = sorted(set(themes_from_files + themes_from_url))
    return list(combined_themes)


def select_theme(name):
    selected_file = name + ".py"
    full_path = os.path.join(gradio_pages_folder, selected_file)

    try:
        with open(config_file, "r", encoding="utf8") as json_file:
            config_data = json.load(json_file)

        if not os.path.exists(full_path):
            # If the theme is from theme_list.json, update only the class field
            config_data["theme"]["file"] = None
            config_data["theme"]["class"] = name
            message = f"Theme {name} successfully selected, restart AllTalk."
        else:
            # If the theme file exists locally, update both file and class fields
            class_found = get_class(full_path)
            if class_found:
                config_data["theme"]["file"] = selected_file
                config_data["theme"]["class"] = class_found
                message = f"Theme {name} successfully selected, restart AllTalk."
            else:
                message = f"Theme {name} was not found."

        with open(config_file, "w", encoding="utf8") as json_file:
            json.dump(config_data, json_file, indent=2)
        
        print(message)
        return message

    except Exception as e:
        print(f"[AllTalk TTS] Error selecting theme {name}: {str(e)}")
        return message

def read_json():
    try:
        with open(config_file, "r", encoding="utf8") as json_file:
            data = json.load(json_file)
            selected_file = data["theme"]["file"]
            class_name = data["theme"]["class"]

            if selected_file and class_name:
                return class_name
            elif not selected_file and class_name:
                return class_name
            else:
                return "gradio/base"
    except Exception as e:
        print(f"[AllTalk TTS] Error reading config.json: {e}")
        return "gradio/base"


def load_json():
    try:
        with open(config_file, "r", encoding="utf8") as json_file:
            data = json.load(json_file)
            selected_file = data["theme"]["file"]
            class_name = data["theme"]["class"]

            if selected_file and class_name:
                module_name = os.path.splitext(selected_file)[0]
                module_spec = importlib.util.spec_from_file_location(module_name, themes_file_path)
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                obtained_class = getattr(module, class_name)
                instance = obtained_class()
                print(f"Theme Loaded: {class_name}")
                return instance
            elif not selected_file and class_name:
                return class_name
            else:
                print("The theme is incorrect.")
                return None
    except Exception as e:
        print(f"Error Loading: {str(e)}")
        return None

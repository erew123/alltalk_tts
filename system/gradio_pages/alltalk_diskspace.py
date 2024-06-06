import os
from pathlib import Path
import gradio as gr

# Define directories
this_dir = Path(__file__).parent.resolve()
main_dir = this_dir.parent.parent.resolve()

# Folder paths
folders = {
    "models": main_dir / "models",
    "finetune": main_dir / "finetune",
    "voices": main_dir / "voices",
    "outputs": main_dir / "outputs",
    "alltalk_environment": main_dir / "alltalk_environment"
}

def get_folder_size_scandir(folder_path):
    total_size = 0
    for entry in os.scandir(folder_path):
        if entry.is_file(follow_symlinks=False):
            try:
                total_size += entry.stat(follow_symlinks=False).st_size
            except OSError as e:
                print(f"Error getting size for file {entry.path}: {e}")
        elif entry.is_dir(follow_symlinks=False):
            try:
                total_size += get_folder_size_scandir(entry.path)
            except OSError as e:
                print(f"Error accessing directory {entry.path}: {e}")
    return total_size

def format_size(size):
    # Convert size to a readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

def calculate_disk_space():
    sizes = {}
    for name, path in folders.items():
        if path.exists():
            folder_size = get_folder_size_scandir(path)
            # print(f"Folder: {name}, Total Size: {folder_size} bytes")
            sizes[name] = format_size(folder_size)
        else:
            sizes[name] = "Not present"
    
    return sizes['models'], sizes['finetune'], sizes['voices'], sizes['outputs'], sizes['alltalk_environment']

def disk_space_page():
    # Define the layout
    with gr.Tab("Disk Space Usage"):
        with gr.Row():
            models_space = gr.Textbox(label="Models Folder", interactive=False)
            finetune_space = gr.Textbox(label="Finetune Folder", interactive=False)
            voices_space = gr.Textbox(label="Voices Folder", interactive=False)
            outputs_space = gr.Textbox(label="Outputs Folder", interactive=False)
            alltalk_env_space = gr.Textbox(label="Alltalk Python Environment", interactive=False)

        calculate_button = gr.Button("Calculate Disk Space Usage")

        # Define actions for buttons
        calculate_button.click(calculate_disk_space, outputs=[models_space, finetune_space, voices_space, outputs_space, alltalk_env_space])

def get_disk_interface():
    return disk_space_page

# Example usage in the main script
if __name__ == "__main__":
    app = gr.Blocks()
    with app:
        disk_space_page()

    app.launch()

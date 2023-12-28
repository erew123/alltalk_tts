import torch
import os
MODEL_PATH = "best_model.pth"
# Check if best_model.pth exists
if not os.path.exists(MODEL_PATH):
    print(f"\nCannot find \033[94m{MODEL_PATH}\033[0m in the current folder.")
    
    # Check if model.pth exists as an alternative
    ALTERNATIVE_PATH = "model.pth"
    if os.path.exists(ALTERNATIVE_PATH):
        print(f"\nIf your model is currently called \033[93m{ALTERNATIVE_PATH}\033[0m please rename it to \033[94mbest_model.pth\033[0m.")
    else:
        print(f"Please put \033[91mbest_model.pth\033[0m in the current folder or rename your model file to \033[94mbest_model.pth\033[0m.")
    exit(1)
# Attempt to load the model
try:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
except Exception as e:
    print("Error loading checkpoint:", e)
    raise
del checkpoint["optimizer"]
for key in list(checkpoint["model"].keys()):
    if "dvae" in key:
        del checkpoint["model"][key]
torch.save(checkpoint, "model.pth")
print(f"\nYour compacted model is created \033[93mmodel.pth\033[0m.")

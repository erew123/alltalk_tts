import os
import json
import pathlib
from random import shuffle

from rvc.configs.config import Config

config = Config()
# Get the RVC directory (parent of configs directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RVC_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def generate_config(rvc_version, sampling_rate, model_path):
    if rvc_version == "v1" or sampling_rate == "40000":
        config_path = f"v1/{sampling_rate}.json"
    else:
        config_path = f"v2/{sampling_rate}.json"
    config_save_path = os.path.join(model_path, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")


def generate_filelist(f0_method, model_path, rvc_version, sampling_rate):
    gt_wavs_dir = f"{model_path}/0_gt_wavs"
    feature_dir = (
        f"{model_path}/3_feature256"
        if rvc_version == "v1"
        else f"{model_path}/3_feature768"
    )
    if f0_method:
        f0_dir = f"{model_path}/2a_f0"
        f0nsf_dir = f"{model_path}/2b-f0nsf"
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    options = []
    for name in names:
        if f0_method:
            options.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0"
            )
        else:
            options.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0")
    fea_dim = 256 if rvc_version == "v1" else 768
    # Add mute files if they exist (optional for training stability)
    mute_wav = f"{RVC_DIR}/logs/mute/0_gt_wavs/mute{sampling_rate}.wav"
    mute_feature = f"{RVC_DIR}/logs/mute/3_feature{fea_dim}/mute.npy"
    mute_f0 = f"{RVC_DIR}/logs/mute/2a_f0/mute.wav.npy"
    mute_f0nsf = f"{RVC_DIR}/logs/mute/2b-f0nsf/mute.wav.npy"

    if f0_method:
        if os.path.exists(mute_wav) and os.path.exists(mute_feature) and os.path.exists(mute_f0) and os.path.exists(mute_f0nsf):
            for _ in range(2):
                options.append(
                    f"{mute_wav}|{mute_feature}|{mute_f0}|{mute_f0nsf}|0"
                )
        else:
            print("Note: Mute files not found, training will proceed without them")
    else:
        if os.path.exists(mute_wav) and os.path.exists(mute_feature):
            for _ in range(2):
                options.append(f"{mute_wav}|{mute_feature}|0")
        else:
            print("Note: Mute files not found, training will proceed without them")
    shuffle(options)
    with open(f"{model_path}/filelist.txt", "w") as f:
        f.write("\n".join(options))
    print(f"Generated filelist with {len(options)} entries")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python preparing_files.py <model_path> <version> <sample_rate> [f0_method]")
        sys.exit(1)

    model_path = sys.argv[1]
    version = sys.argv[2]
    sample_rate = sys.argv[3]
    # Default to True for f0 (pitch-based training)
    f0_method = True if len(sys.argv) < 5 else sys.argv[4].lower() in ("true", "1", "yes")

    print(f"Preparing files for {model_path}")
    print(f"Version: {version}, Sample rate: {sample_rate}, F0 method: {f0_method}")

    # Generate config file
    generate_config(version, sample_rate, model_path)
    print("Config generated")

    # Generate filelist
    generate_filelist(f0_method, model_path, version, sample_rate)
    print("Filelist generation complete")

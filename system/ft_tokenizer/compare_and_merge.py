import json
import os

# Define file paths
base_path = "/alltalk_tts/system/ft_tokenizer/"
original_file = os.path.join(base_path, "/alltalk_tts/models/xtts/xttsv2_2.0.3/vocab.json") # base model vocab.json
new_file = os.path.join(base_path, "/expanded_model/bpe_tokenizer-vocab.json")  # new bpe tokenizer vocab.json
output_file = os.path.join(base_path, "/expanded_model/expanded_vocab.json")  # output expanded_vocab.json file/path


with open(original_file, 'r') as f:
    original_data = json.load(f)

with open(new_file, 'r') as f:
    new_data = json.load(f)

# Get the original and new vocabularies
original_vocab = original_data['model']['vocab']
new_vocab = new_data['model']['vocab']

# Find the maximum value in the original vocabulary. We do this so we make sure to append and not overwrite any new values
max_value = max(original_vocab.values())

# merge
for key, value in new_vocab.items():
    if key not in original_vocab:
        max_value += 1
        original_vocab[key] = max_value

# Update
original_data['model']['vocab'] = original_vocab

# Get the original and new merges
original_merges = original_data['model']['merges']
new_merges = new_data['model']['merges']

# Merge the merges
merged_merges = original_merges.copy()
for merge in new_merges:
    if merge not in merged_merges:
        merged_merges.append(merge)

# Update
original_data['model']['merges'] = merged_merges
with open(output_file, 'w') as f:
    json.dump(original_data, f, ensure_ascii=False, indent=2)

print(f"Merged vocabulary and merges saved to {output_file}")

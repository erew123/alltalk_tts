import json
import os

def merge_vocabularies(base_vocab_path, new_vocab_path, output_path):
    # Load the base model's vocab.json
    with open(base_vocab_path, 'r') as f:
        base_data = json.load(f)

    # Load the new bpe_tokenizer.json
    with open(new_vocab_path, 'r') as f:
        new_data = json.load(f)

    # Extract the vocabularies
    base_vocab = base_data['model']['vocab']
    new_vocab = new_data['model']['vocab']

    # Find the maximum value in the base vocabulary
    max_value = max(base_vocab.values())

    # Merge the vocabularies
    for key, value in new_vocab.items():
        if key not in base_vocab:
            max_value += 1
            base_vocab[key] = max_value

    # Update the base data with the merged vocabulary
    base_data['model']['vocab'] = base_vocab

    # Extract the merges
    base_merges = base_data['model']['merges']
    new_merges = new_data['model']['merges']

    # Merge the merges
    merged_merges = base_merges.copy()
    for merge in new_merges:
        if merge not in merged_merges:
            merged_merges.append(merge)

    # Update the base data with the merged merges
    base_data['model']['merges'] = merged_merges

    # Write the merged vocabulary and merges to the output file
    with open(output_path, 'w') as f:
        json.dump(base_data, f, ensure_ascii=False, indent=2)

    print(f"Merged vocabulary and merges saved to {output_path}")

# Define file paths
base_vocab_path = "/alltalk_tts/models/xtts/xttsv2_2.0.2/originalvocab.json" # base model vocab.json path (2.0.2)
new_vocab_path = "/alltalk_tts/expanded_models/tortoise_tokenizer2xttsv2.json" # path to the custom dataset vocab.json
output_path = "/alltalk_tts/expanded_models/combined_vocab.json" # location for combined vocab.json

# Merge the vocabularies
merge_vocabularies(base_vocab_path, new_vocab_path, output_path)
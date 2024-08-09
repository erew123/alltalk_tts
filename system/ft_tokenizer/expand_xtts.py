"""
This script does not have any specific alltalk integration yet. So we are manually entering paths and it works as a standalone.

The script does the following:
    - Expands the embedding layer of the base XTTSv2 model according to the user created/trained bpe_tokenizer-vocab.json and the base model vocab.json.
Set variable paths with the base config and base model.pth. 
Set the new tokenizer/vocab bpe_tokenizer-vocab.json location.
The new model will be saved at \expanded_models\expanded_model.pth

Once this is done the new expanded model must be swapped in for the base model.pth then combined with the dvae.pth, vocab.json(bpe_tokenizer-vocab.json), base model config.json,
base model speaker_xtts.pth and base model vocoder.json.


I left some print debug statements in the script, they may be nice for the user to see during the process.

"""

import torch
import torch.nn as nn
import json
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainerConfig
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

config_path = "/alltalk_tts/models/xtts/xttsv2_2.0.3/config.json"                        # Path to the base model config.json
pretrained_model_path = "/alltalk_tts/models/xtts/xttsv2_2.0.3/model.pth"                # Path to the base model.pth
new_tokenizer_path = "/expanded_model/expanded_vocab.json"                               # Path to the new combined expanded_vocab.json
expanded_model_path = "/expanded_model/expanded_model.pth"                               # Path to where you want the new expanded_model.pth


# Open and load the configuration file
with open(config_path, "r") as f:
    config_dict = json.load(f)

# Create a GPTTrainerConfig object and populate it with the loaded configuration
config = GPTTrainerConfig()
config.from_dict(data=config_dict)

# Function to get the vocabulary size from a tokenizer file
def get_vocab_size(tokenizer_path):
    tokenizer = VoiceBpeTokenizer(vocab_file=tokenizer_path)
    return len(tokenizer.tokenizer.get_vocab())

# Function to adjust the pretrained model with a new tokenizer
def adjust_pretrained_model(
    pretrained_model_path, adjusted_model_path, new_tokenizer_path
):
    # Load the pretrained model state dictionary
    state_dict = torch.load(pretrained_model_path)
    pretrained_state_dict = state_dict["model"]

    # Create a new Xtts model instance with the loaded configuration
    model = Xtts(config)

    # Load the pretrained state dictionary into the new model
    missing_keys, unexpected_keys = model.load_state_dict(
        pretrained_state_dict, strict=False
    )
    # Print any missing or unexpected keys for debugging
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    print("Pretrained model loaded successfully.")

    # Create a new tokenizer with the new vocabulary
    new_tokenizer = VoiceBpeTokenizer(vocab_file=new_tokenizer_path)

    # Get the old and new vocabulary sizes, and the embedding dimension
    old_vocab_size = model.gpt.text_embedding.num_embeddings
    new_vocab_size = len(new_tokenizer.tokenizer.get_vocab())
    embedding_dim = model.gpt.text_embedding.embedding_dim

    # Print vocabulary sizes and embedding dimension for debugging
    print(f"Old vocab size: {old_vocab_size}")
    print(f"New vocab size: {new_vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")

    # Adjust the embedding layer with the new vocabulary size
    adjust_embedding_layer(
        model, pretrained_state_dict, new_vocab_size, adjusted_model_path
    )

# Function to adjust the embedding layer for the new vocabulary size
def adjust_embedding_layer(
    model, pretrained_state_dict, new_vocab_size, adjusted_model_path
):
    old_vocab_size = model.gpt.text_embedding.num_embeddings
    embedding_dim = model.gpt.text_embedding.embedding_dim

    # Create new embedding and linear layers with the new vocabulary size
    new_text_embedding = nn.Embedding(new_vocab_size, embedding_dim)
    new_text_head = nn.Linear(embedding_dim, new_vocab_size)

    # Copy weights from the old embedding layer to the new one
    if new_vocab_size > old_vocab_size:
        # If the new vocabulary is larger, copy existing weights and initialize new ones
        new_text_embedding.weight.data[:old_vocab_size] = (
            model.gpt.text_embedding.weight.data
        )
        new_text_head.weight.data[:old_vocab_size] = model.gpt.text_head.weight.data
        new_text_head.bias.data[:old_vocab_size] = model.gpt.text_head.bias.data

        # Initialize new weights with normal distribution
        new_text_embedding.weight.data[old_vocab_size:].normal_(mean=0.0, std=0.02)
        new_text_head.weight.data[old_vocab_size:].normal_(mean=0.0, std=0.02)
        new_text_head.bias.data[old_vocab_size:].normal_(mean=0.0, std=0.02)
    else:
        # If the new vocabulary is smaller, truncate the existing weights
        new_text_embedding.weight.data = model.gpt.text_embedding.weight.data[
            :new_vocab_size
        ]
        new_text_head.weight.data = model.gpt.text_head.weight.data[:new_vocab_size]
        new_text_head.bias.data = model.gpt.text_head.bias.data[:new_vocab_size]

    # Replace the old embedding layer with the new one
    model.gpt.text_embedding = new_text_embedding
    model.gpt.text_head = new_text_head

    # Save the adjusted model
    checkpoint = {"model": model.state_dict()}
    torch.save(checkpoint, adjusted_model_path)
    print(f"Adjusted model saved to {adjusted_model_path}")

# Expand the pretrained model with the new tokenizer
adjust_pretrained_model(pretrained_model_path, expanded_model_path, new_tokenizer_path)

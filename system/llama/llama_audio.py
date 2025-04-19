#############################################################################################################
#
#   This Llama audio implementation is from Orpheus-FastAPI (https://github.com/Lex-au/Orpheus-FastAPI)
#   See: inference.py (https://github.com/Lex-au/Orpheus-FastAPI/blob/main/tts_engine/inference.py)
#
#   Modified into class format
#   Changed token decoding into asynchronous task
#
#############################################################################################################


import asyncio
import json
import os
from pathlib import Path
import time
from typing import Generator
import wave
from llama_cpp import Llama
import numpy as np
import torch
from snac import SNAC

from system.proxy_module.twisted_server import print_message

SAMPLE_RATE = 24000

# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"
MAX_CACHE_SIZE = 10000  # Increased cache size for better performance

# Check if CUDA is available and set device accordingly
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

cuda_stream = None
if snac_device == "cuda":
    cuda_stream = torch.cuda.Stream()

class LlamaAudio:



    def __init__(self, **kwargs):
        """
        Initializes the LLaMA model with any parameters supported by llama_cpp.Llama.
        """

        if "model_path" not in kwargs:
            raise ValueError("You must provide a 'model_path' to initialize the Llama model.")
        
        self.llama = Llama(**kwargs)

        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self.model = self.model.to(snac_device)

        # Define the path to the confignew.json file
        configfile_path = Path(__file__).parent.parent.parent.resolve() / "confignew.json"
        # Load config file and get settings
        with open(configfile_path, "r") as configfile:
            configfile_data = json.load(configfile)
        self.debug_tts = configfile_data.get("debugging").get("debug_tts")
        self.debug_variables = configfile_data.get("debugging").get("debug_tts_variables")

    
    def generate(self, text, output_file, **kwargs):
        result = self.generate_tokens(text, **kwargs)
        return self.tokens_decoder_async(result, output_file)

        


    def generate_tokens(self, text, **kwargs):
        token_counter = 0
        start_time = time.time()

        if self.debug_variables:
            print_message("Llama parameters: ", "debug_tts_variables", "LLAMA")
            keys = list(kwargs.keys())
            for i, key in enumerate(keys):
                # Use └─ for the last item, and ├─ for others
                prefix = "└─" if i == len(keys) - 1 else "├─"
                print_message(f"{prefix} {key}: {kwargs[key]}", "debug_tts_variables", "LLAMA")

        for data in self.llama.create_completion(text, **kwargs, stream=True):

            try:
                if 'choices' in data and len(data['choices']) > 0:
                    token_chunk = data['choices'][0].get('text', '')
                    for token_text in token_chunk.split('>'):
                        token_text = f'{token_text}>'
                        token_counter += 1

                        if token_text:
                            yield token_text

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue            
                
        generation_time = time.time() - start_time
        tokens_per_second = token_counter / generation_time if generation_time > 0 else 0

        if self.debug_tts:
            print_message(f"Token generation complete: {token_counter} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)", "debug_tts", "LLAMA")
        return
    


    async def tokens_decoder_async(self, syn_token_gen, output_file=None):
        """Async wrapper with asyncio-compatible queue and task scheduling."""
        queue_size = 100 #if HIGH_END_GPU else 50
        audio_queue = asyncio.Queue(maxsize=queue_size)
        audio_segments = []

        # Prepare WAV file if needed
        wav_file = None
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            wav_file = wave.open(str(output_file), "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)

        batch_size = 32 #if HIGH_END_GPU else 16
        buffer_max_size = 1024 * 1024  # 1MB
        write_buffer = bytearray()

        async def async_token_gen():
            batch = []
            for token in syn_token_gen:
                batch.append(token)
                if len(batch) >= batch_size:
                    for t in batch:
                        yield t
                    batch = []
            for t in batch:
                yield t

        async def producer():
            chunk_count = 0
            last_log_time = time.time()
            try:
                async for audio_chunk in self.tokens_decoder(async_token_gen()):
                    if audio_chunk:
                        await audio_queue.put(audio_chunk)
                        chunk_count += 1
                        current_time = time.time()
                        if current_time - last_log_time >= 3.0:
                            elapsed = current_time - last_log_time
                            if (self.debug_tts):
                                print_message(f"Audio generation rate: {chunk_count / elapsed:.2f} chunks/second", "debug_tts", "LLAMA")
                            last_log_time = current_time
                            chunk_count = 0
            except Exception as e:
                print(f"Error in token processing: {e}")
            finally:
                await audio_queue.put(None)  # Sentinel

        async def consumer():
            last_flush_time = time.time()
            while True:
                audio = await audio_queue.get()
                if audio is None:
                    break
                audio_segments.append(audio)
                if wav_file:
                    write_buffer.extend(audio)
                    if len(write_buffer) >= buffer_max_size:
                        wav_file.writeframes(write_buffer)
                        write_buffer.clear()
                # Periodic flush
                if wav_file and (time.time() - last_flush_time > 1.0):
                    wav_file.writeframes(write_buffer)
                    write_buffer.clear()
                    last_flush_time = time.time()

            # Final flush
            if wav_file and len(write_buffer) > 0:
                wav_file.writeframes(write_buffer)

        await asyncio.gather(producer(), consumer())

        if wav_file:
            wav_file.close()
            if self.debug_tts:
                print_message(f"Audio saved to {output_file}", "debug_tts", "LLAMA")

        return audio_segments
    



    async def tokens_decoder(self, token_gen) -> Generator[bytes, None, None]:
        """Simplified token decoder with early first-chunk processing for lower latency."""
        buffer = []
        count = 0

        # Use different thresholds for first chunk vs. subsequent chunks
        first_chunk_processed = False
        min_frames_first = 7  # Process after just 7 tokens for first chunk (ultra-low latency)
        min_frames_subsequent = 28  # Default for reliability after first chunk (4 chunks of 7)
        process_every = 7  # Process every 7 tokens (standard for Orpheus model)

        start_time = time.time()
        last_log_time = start_time
        token_count = 0

        async for token_text in token_gen:
            token = self.turn_token_into_id(token_text, count)
            if token is not None and token > 0:
                # Add to buffer using simple append (reliable method)
                buffer.append(token)
                count += 1
                token_count += 1

                # Log throughput periodically
                current_time = time.time()
                if current_time - last_log_time > 5.0:  # Every 5 seconds
                    elapsed = current_time - start_time
                    if self.debug_tts and elapsed > 0:
                        print_message(f"Token processing rate: {token_count/elapsed:.1f} tokens/second", "debug_tts", "LLAMA")
                    last_log_time = current_time

                # Different processing paths based on whether first chunk has been processed
                if not first_chunk_processed:
                    # For first audio output, process as soon as we have enough tokens for one chunk
                    if count >= min_frames_first:
                        buffer_to_proc = buffer[-min_frames_first:]

                        # Process the first chunk for immediate audio feedback
                        if self.debug_tts:
                            print_message(f"Processing first audio chunk with {len(buffer_to_proc)} tokens", "debug_tts", "LLAMA")

                        audio_samples = self.convert_to_audio(buffer_to_proc, count)
                        if audio_samples is not None:
                            first_chunk_processed = True  # Mark first chunk as processed
                            yield audio_samples
                else:
                    # For subsequent chunks, use standard processing with larger batch
                    if count % process_every == 0 and count >= min_frames_subsequent:
                        # Use simple slice operation - reliable and correct
                        buffer_to_proc = buffer[-min_frames_subsequent:]

                        # Debug output to help diagnose issues
                        if self.debug_tts and count % 28 == 0:
                            print_message(f"Processing buffer with {len(buffer_to_proc)} tokens, total collected: {len(buffer)}", "debug_tts", "LLAMA")

                        # Process the tokens
                        audio_samples = self.convert_to_audio(buffer_to_proc, count)
                        if audio_samples is not None:
                            yield audio_samples


    # Use a single global cache for token processing
    token_id_cache = {}

    def turn_token_into_id(self, token_string, index):
        """
        Optimized token-to-ID conversion with caching.
        This is the definitive implementation used by both inference.py and speechpipe.py.

        Args:
            token_string: The token string to convert
            index: Position index used for token offset calculation

        Returns:
            int: Token ID if valid, None otherwise
        """
        # Check cache first (significant speedup for repeated tokens)
        cache_key = (token_string, index % 7)
        if cache_key in self.token_id_cache:
            return self.token_id_cache[cache_key]

        # Early rejection for obvious non-matches
        if CUSTOM_TOKEN_PREFIX not in token_string:
            return None

        # Process token
        token_string = token_string.strip()
        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

        if last_token_start == -1:
            return None

        last_token = token_string[last_token_start:]

        if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
            return None

        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)

            # Cache the result if it's valid
            if len(self.token_id_cache) < MAX_CACHE_SIZE:
                self.token_id_cache[cache_key] = token_id

            return token_id
        except (ValueError, IndexError):
            return None
        


    def convert_to_audio(self, multiframe, count):
        """
        Optimized version of convert_to_audio that eliminates inefficient tensor operations
        and reduces CPU-GPU transfers for much faster inference on high-end GPUs.
        """
        if len(multiframe) < 7:
            return None
    
        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames*7]

        # Pre-allocate tensors instead of incrementally building them
        codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
        codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
        codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)

        # Use vectorized operations where possible
        frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)

        # Direct indexing is much faster than concatenation in a loop
        for j in range(num_frames):
            idx = j * 7

            # Code 0 - single value per frame
            codes_0[j] = frame_tensor[idx]

            # Code 1 - two values per frame
            codes_1[j*2] = frame_tensor[idx+1]
            codes_1[j*2+1] = frame_tensor[idx+4]

            # Code 2 - four values per frame
            codes_2[j*4] = frame_tensor[idx+2]
            codes_2[j*4+1] = frame_tensor[idx+3]
            codes_2[j*4+2] = frame_tensor[idx+5]
            codes_2[j*4+3] = frame_tensor[idx+6]

        # Reshape codes into expected format
        codes = [
            codes_0.unsqueeze(0), 
            codes_1.unsqueeze(0), 
            codes_2.unsqueeze(0)
        ]

        # Check tokens are in valid range
        if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
            torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
            torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
            return None

        # Use CUDA stream for parallel processing if available
        stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()

        with stream_ctx, torch.inference_mode():
            # Decode the audio
            audio_hat = self.model.decode(codes)

            # Extract the relevant slice and efficiently convert to bytes
            # Keep data on GPU as long as possible
            audio_slice = audio_hat[:, :, 2048:4096]

            # Process on GPU if possible, with minimal data transfer
            if snac_device == "cuda":
                # Scale directly on GPU
                audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
                # Only transfer the final result to CPU
                audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
            else:
                # For non-CUDA devices, fall back to the original approach
                detached_audio = audio_slice.detach().cpu()
                audio_np = detached_audio.numpy()
                audio_int16 = (audio_np * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

        return audio_bytes
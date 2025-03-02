import torch
import soundfile as sf
import torchaudio
from transformers import pipeline


def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def text_to_speech(llm, tokenizer, codec_model, sample_audio_path, target_text, temperature, repetition_penalty, device, prompt_text=None):
    # only 16khz speech support!
    prompt_wav, sr = sf.read(sample_audio_path)
    prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)
    print(prompt_wav.shape)

    # if prompt_text is None:
    #     prompt_text = whisper_turbo_pipe(sample_audio_path)['text'].strip()

    input_text = prompt_text + ' ' + target_text

    with torch.no_grad():
        # Encode the prompt wav
        vq_code_prompt = codec_model.encode_code(input_waveform=prompt_wav)
        print("Prompt Vq Code Shape:", vq_code_prompt.shape )

        vq_code_prompt = vq_code_prompt[0,0,:]
        # Convert int 12345 to token <|s_12345|>
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

        # Tokenize the text and the speech prefix
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
        ]

        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True
        )
        input_ids = input_ids.to(device)
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        # Generate the speech autoregressively
        outputs = llm.generate(
            input_ids,
            max_length=2048, # Pretrained model max length
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=1,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )

        generated_ids = outputs[0][input_ids.shape[1]:-1]

        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Convert  token <|s_23456|> to int 23456
        speech_tokens = extract_speech_ids(speech_tokens)

        speech_tokens = torch.tensor(speech_tokens).to(device).unsqueeze(0).unsqueeze(0)

        # Decode the speech tokens to speech waveform
        gen_wav = codec_model.decode_code(speech_tokens)

    print(gen_wav.shape)
    return gen_wav[0, 0, :].to("cpu").numpy()

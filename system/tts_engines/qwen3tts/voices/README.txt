This folder is only used when a Base-variant model is loaded
(Qwen3-TTS-12Hz-0.6B-Base or Qwen3-TTS-12Hz-1.7B-Base), for 3-second
reference-audio voice cloning.

To add a cloning voice, drop in a matching pair of files with the same
basename:

    my_voice.wav
    my_voice.reference.txt

- my_voice.wav is a short (roughly 3-10 second), clean, single-speaker
  reference recording.
- my_voice.reference.txt contains the exact transcript of what is spoken
  in that wav file, as plain text.

The wav's basename (without extension) becomes the voice name shown in
AllTalk.

CustomVoice and VoiceDesign models do NOT use this folder -- their voices
are the Custom Voice Presets / Voice Design Presets managed from this
engine's settings page (backed by ../qwen3tts_voices.json).

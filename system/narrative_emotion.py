"""
Narrative Emotion Detection for TTS

This module provides automatic emotion detection from narrative text,
extracting speech cues like "she whispered angrily" and applying
appropriate emotion markers to TTS output.

Supports:
- Fish Speech: Applies (emotion) markers directly
- Parler TTS: Modifies voice descriptions dynamically

Usage:
    from narrative_emotion import NarrativeEmotionDetector

    detector = NarrativeEmotionDetector(config)
    processed_parts = detector.process_segments(text_segments, engine_type)
"""

import re
import json
import requests
from typing import List, Tuple, Optional, Set
from pathlib import Path

from .narrative_patterns import (
    COMPILED_PATTERNS,
    get_valid_marker,
    FISH_SPEECH_MARKERS,
)


class NarrativeEmotionDetector:
    """
    Detects emotions from narrative text and applies appropriate markers
    for TTS generation.
    """

    # Parler voice description modifiers for each emotion
    PARLER_EMOTION_DESCRIPTIONS = {
        'whispering': 'whispering softly in a hushed, intimate tone',
        'shouting': 'speaking loudly with an intense, raised voice',
        'screaming': 'screaming with extreme intensity',
        'angry': 'speaking with an angry, aggressive tone',
        'furious': 'speaking with intense fury and rage',
        'sad': 'speaking sadly with a melancholic, somber tone',
        'depressed': 'speaking in a deeply sad, heavy tone',
        'excited': 'speaking excitedly with enthusiasm and energy',
        'joyful': 'speaking joyfully with happiness in the voice',
        'nervous': 'speaking nervously with a shaky, uncertain voice',
        'anxious': 'speaking anxiously with worry in the voice',
        'scared': 'speaking fearfully with a trembling voice',
        'surprised': 'speaking with surprise and astonishment',
        'astonished': 'speaking with complete amazement',
        'sarcastic': 'speaking sarcastically with a mocking undertone',
        'serious': 'speaking seriously with a firm, grave tone',
        'confident': 'speaking confidently with authority',
        'proud': 'speaking proudly with self-assurance',
        'embarrassed': 'speaking sheepishly with embarrassment',
        'hesitating': 'speaking hesitantly with uncertainty',
        'soft tone': 'speaking softly and gently',
        'in a hurry tone': 'speaking quickly and urgently',
        'laughing': 'speaking while laughing',
        'sighing': 'speaking with a sigh',
        'sobbing': 'speaking while crying softly',
        'crying loudly': 'speaking while crying intensely',
        'panting': 'speaking breathlessly',
    }

    # Orpheus TTS uses <tag> format for vocal effects
    # Maps detected emotions to Orpheus tags
    ORPHEUS_EMOTION_TAGS = {
        # Emotion tags (speaking style)
        'happy': '<happy>',
        'sad': '<sad>',
        'angry': '<angry>',
        'frustrated': '<frustrated>',
        'excited': '<excited>',
        'curious': '<curious>',
        'surprised': '<surprise>',
        'surprise': '<surprise>',
        'disgusted': '<disgust>',
        'disgust': '<disgust>',
        'crying': '<crying>',
        'sleepy': '<sleepy>',
        'panicky': '<panicky>',
        'panic': '<panicky>',
        'shouting': '<shout>',
        'shout': '<shout>',
        'yelling': '<shout>',
        'whispering': '<whisper>',
        'whisper': '<whisper>',
        'soft': '<whisper>',
        'soft tone': '<whisper>',
        'softened': '<whisper>',
        'gentle': '<whisper>',
        'fast': '<fast>',
        'slow': '<slow>',
        'deep': '<deep>',
        'high': '<high>',
        'longer': '<longer>',
        'normal': '<normal>',
        # Emotive tags (sound effects)
        'laughing': '<laugh>',
        'chuckling': '<chuckle>',
        'sighing': '<sigh>',
        'coughing': '<cough>',
        'sniffling': '<sniffle>',
        'groaning': '<groan>',
        'yawning': '<yawn>',
        'gasping': '<gasp>',
        # Emotion to tag mappings
        'nervous': '<panicky>',
        'anxious': '<panicky>',
        'joyful': '<happy>',
        'amused': '<chuckle>',
        'tired': '<sleepy>',
        'exhausted': '<yawn>',
        'fearful': '<panicky>',
        'scared': '<panicky>',
        'serious': '<deep>',
        'sobbing': '<sniffle>',
        'crying loudly': '<crying>',
        'panting': '<gasp>',
    }

    # Valid Orpheus emotion tags (speaking style)
    ORPHEUS_EMOTION_VALID = {
        '<happy>', '<normal>', '<disgust>', '<longer>', '<sad>', '<frustrated>',
        '<slow>', '<excited>', '<whisper>', '<panicky>', '<curious>', '<surprise>',
        '<fast>', '<crying>', '<deep>', '<sleepy>', '<angry>', '<high>', '<shout>'
    }

    # Valid Orpheus emotive tags (sound effects)
    ORPHEUS_EMOTIVE_VALID = {
        '<laugh>', '<chuckle>', '<sigh>', '<cough>', '<sniffle>', '<groan>', '<yawn>', '<gasp>'
    }

    def __init__(self, config=None):
        """
        Initialize the detector.

        Args:
            config: Configuration object with narrative_emotion settings
        """
        self.config = config
        self.detection_mode = "basic"  # "basic" or "compound"
        self.use_llm = False
        self.llm_api_url = "http://127.0.0.1:11434"
        self.llm_api_key = ""
        self.llm_model = "llama3.2"
        self.apply_to_narrator = False
        self.apply_to_character = True
        self.apply_to_ambiguous = True

        if config and hasattr(config, 'narrative_emotion'):
            ne = config.narrative_emotion
            self.detection_mode = getattr(ne, 'detection_mode', 'basic')
            self.use_llm = getattr(ne, 'use_llm_inference', False)
            self.llm_api_url = getattr(ne, 'llm_api_url', 'http://127.0.0.1:11434')
            self.llm_api_key = getattr(ne, 'llm_api_key', '')
            self.llm_model = getattr(ne, 'llm_model', 'llama3.2')
            self.apply_to_narrator = getattr(ne, 'apply_to_narrator', False)
            self.apply_to_character = getattr(ne, 'apply_to_character', True)
            self.apply_to_ambiguous = getattr(ne, 'apply_to_ambiguous', True)

    def detect_emotions(self, text: str) -> Set[str]:
        """
        Detect emotions from a text segment using pattern matching.

        Args:
            text: The text to analyze (typically narrator text around dialogue)

        Returns:
            Set of detected emotion markers
        """
        emotions = set()

        # Check speech verbs
        for pattern, emotion in COMPILED_PATTERNS['speech_verbs']:
            if pattern.search(text):
                marker = get_valid_marker(emotion)
                if marker:
                    emotions.add(marker)

        # Check emotion adverbs
        for pattern, emotion in COMPILED_PATTERNS['emotion_adverbs']:
            if pattern.search(text):
                marker = get_valid_marker(emotion)
                if marker:
                    emotions.add(marker)

        # Check context phrases
        for pattern, emotion in COMPILED_PATTERNS['context_phrases']:
            if pattern.search(text):
                marker = get_valid_marker(emotion)
                if marker:
                    emotions.add(marker)

        return emotions

    def detect_compound_emotions(self, text: str) -> List[Tuple[int, str]]:
        """
        Detect emotion transitions/changes within text.

        Looks for patterns like "but then softened", "growing angrier", etc.
        and returns positions where emotion markers should be inserted.

        Args:
            text: The text to analyze

        Returns:
            List of (character_position, emotion) tuples sorted by position
        """
        emotion_positions = []

        # Patterns for emotion transitions
        transition_patterns = [
            # "but then [emotion]" patterns
            (r'\b(but then|then)\s+(softened|calmed down|relaxed)', 'soft tone'),
            (r'\b(but then|then)\s+(grew angry|got angry|became furious)', 'angry'),
            (r'\b(but then|then)\s+(smiled|laughed|chuckled)', 'laughing'),
            (r'\b(but then|then)\s+(sighed|exhaled)', 'sighing'),
            (r'\b(but then|then)\s+(cried|sobbed|teared up)', 'sobbing'),
            (r'\b(but then|then)\s+(whispered|lowered .* voice)', 'whispering'),
            (r'\b(but then|then)\s+(shouted|yelled|screamed)', 'shouting'),
            # "growing/becoming [emotion]" patterns
            (r'\b(growing|becoming|getting)\s+(angrier|more angry|furious)', 'angry'),
            (r'\b(growing|becoming|getting)\s+(sadder|more sad|depressed)', 'sad'),
            (r'\b(growing|becoming|getting)\s+(excited|enthusiastic)', 'excited'),
            (r'\b(growing|becoming|getting)\s+(nervous|anxious)', 'nervous'),
            (r'\b(growing|becoming|getting)\s+(quieter|softer)', 'whispering'),
            (r'\b(growing|becoming|getting)\s+(louder|more intense)', 'shouting'),
            # "voice [changed]" patterns
            (r'\bvoice\s+(softened|dropped|lowered)', 'whispering'),
            (r'\bvoice\s+(rose|raised|crescendoed)', 'shouting'),
            (r'\bvoice\s+(cracked|broke|trembled)', 'nervous'),
            (r'\bvoice\s+(hardened|turned cold|grew stern)', 'serious'),
            # Sentence-level emotion indicators (typically mid-sentence)
            (r'[,;]\s*(\w+\s+)?(laughing|chuckling)', 'laughing'),
            (r'[,;]\s*(\w+\s+)?(sighing|with a sigh)', 'sighing'),
            (r'[,;]\s*(\w+\s+)?(crying|sobbing)', 'sobbing'),
            (r'[,;]\s*(\w+\s+)?(whispering|in a whisper)', 'whispering'),
            (r'[,;]\s*(\w+\s+)?(shouting|yelling)', 'shouting'),
        ]

        for pattern_str, emotion in transition_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                # Insert emotion marker after the transition phrase
                position = match.end()
                marker = get_valid_marker(emotion)
                if marker:
                    emotion_positions.append((position, marker))

        # Sort by position
        emotion_positions.sort(key=lambda x: x[0])

        return emotion_positions

    def _llm_infer_emotion(self, narrator_text: str, dialogue_text: str, engine_type: str = 'fishspeech') -> Set[str]:
        """
        Use LLM to infer emotion when no explicit cues are found.

        Args:
            narrator_text: The surrounding narrative context
            dialogue_text: The actual dialogue being spoken
            engine_type: The TTS engine type for appropriate emotion format

        Returns:
            Set of inferred emotion markers
        """
        if not self.use_llm or not self.llm_model:
            return set()

        try:
            # Build engine-specific valid emotions list
            if engine_type == 'orpheus':
                valid_emotions = list(self.ORPHEUS_EMOTION_TAGS.keys())
            elif engine_type == 'fishspeech':
                valid_emotions = list(FISH_SPEECH_MARKERS)
            else:
                valid_emotions = ['neutral', 'angry', 'sad', 'excited', 'whispering', 'shouting', 'nervous', 'laughing', 'surprised', 'serious', 'sarcastic']

            prompt = f"""Analyze this narrative text and determine the emotion for the dialogue.

Narrator context: {narrator_text}
Dialogue: {dialogue_text}

What emotion should the dialogue be spoken with?
Choose ONE from: {', '.join(valid_emotions)}
Answer with just ONE word (the emotion):"""

            # Call LLM API (OpenAI-compatible format)
            result = self._call_llm_api(prompt)
            if result:
                # Clean and validate the response
                emotion = result.strip().lower().replace('.', '').replace(',', '')
                # Check if it's a valid emotion
                marker = get_valid_marker(emotion)
                if marker:
                    return {marker}
                # For orpheus, check direct mapping
                if engine_type == 'orpheus' and emotion in self.ORPHEUS_EMOTION_TAGS:
                    return {emotion}
            return set()

        except Exception as e:
            print(f"[NarrativeEmotion] LLM inference failed: {e}")
            return set()

    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """
        Call the LLM API with the given prompt.

        Supports both OpenAI-compatible APIs and Ollama.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response text or None on failure
        """
        try:
            headers = {"Content-Type": "application/json"}
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"

            # Detect API type from URL
            api_url = self.llm_api_url.rstrip('/')

            if 'ollama' in api_url.lower() or ':11434' in api_url:
                # Ollama API format
                endpoint = f"{api_url}/api/generate"
                payload = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 20
                    }
                }
                response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                return response.json().get('response', '')

            else:
                # OpenAI-compatible API format
                endpoint = f"{api_url}/v1/chat/completions"
                payload = {
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": "You are an emotion analyzer. Respond with only ONE word - the detected emotion."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 20
                }
                response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']

        except requests.exceptions.Timeout:
            print("[NarrativeEmotion] LLM API timeout")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[NarrativeEmotion] LLM API request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[NarrativeEmotion] LLM API response parse error: {e}")
            return None

    def apply_fish_speech_markers(self, text: str, emotions: Set[str]) -> str:
        """
        Apply emotion markers in Fish Speech format.

        Args:
            text: The dialogue text
            emotions: Set of detected emotions

        Returns:
            Text with emotion markers prepended
        """
        if not emotions:
            return text

        # Filter to valid Fish Speech markers
        valid_emotions = [e for e in emotions if e in FISH_SPEECH_MARKERS]

        if not valid_emotions:
            return text

        # Prepend markers
        markers = ' '.join(f'({e})' for e in valid_emotions)
        return f"{markers} {text}"

    def apply_parler_description(self, base_description: str, emotions: Set[str]) -> str:
        """
        Modify Parler voice description to include emotion.

        Args:
            base_description: The original voice description
            emotions: Set of detected emotions

        Returns:
            Modified voice description
        """
        if not emotions:
            return base_description

        # Get the first emotion's description modifier
        for emotion in emotions:
            if emotion in self.PARLER_EMOTION_DESCRIPTIONS:
                modifier = self.PARLER_EMOTION_DESCRIPTIONS[emotion]
                # Append to base description
                # Remove trailing period if present, add modifier
                base = base_description.rstrip('.')
                return f"{base}, {modifier}."

        return base_description

    def apply_orpheus_markers(self, text: str, emotions: Set[str]) -> str:
        """
        Apply emotion markers in Orpheus format (<tag>).

        Orpheus supports: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>

        Args:
            text: The dialogue text
            emotions: Set of detected emotions

        Returns:
            Text with Orpheus tags inserted
        """
        if not emotions:
            return text

        # Collect unique Orpheus tags for detected emotions
        tags = set()
        for emotion in emotions:
            if emotion in self.ORPHEUS_EMOTION_TAGS:
                tags.add(self.ORPHEUS_EMOTION_TAGS[emotion])

        if not tags:
            return text

        # Prepend tags to text (Orpheus can handle multiple tags)
        tag_str = ' '.join(sorted(tags))
        return f"{tag_str} {text}"

    def apply_compound_emotions(self, text: str, engine_type: str = 'orpheus') -> str:
        """
        Apply compound emotions with mid-text markers.

        Detects emotion transitions and inserts markers at appropriate positions.

        Args:
            text: The text to process
            engine_type: 'orpheus', 'fishspeech', or 'parler'

        Returns:
            Text with emotion markers inserted at transition points
        """
        # First get initial emotions
        initial_emotions = self.detect_emotions(text)

        # Get compound emotion positions
        compound_positions = self.detect_compound_emotions(text)

        # Build result with markers inserted
        result = text
        offset = 0  # Track offset as we insert markers

        # Apply initial emotions at the start
        if initial_emotions:
            if engine_type == 'orpheus':
                tags = set()
                for emotion in initial_emotions:
                    if emotion in self.ORPHEUS_EMOTION_TAGS:
                        tags.add(self.ORPHEUS_EMOTION_TAGS[emotion])
                if tags:
                    prefix = ' '.join(sorted(tags)) + ' '
                    result = prefix + result
                    offset += len(prefix)
            elif engine_type == 'fishspeech':
                valid_emotions = [e for e in initial_emotions if e in FISH_SPEECH_MARKERS]
                if valid_emotions:
                    prefix = ' '.join(f'({e})' for e in valid_emotions) + ' '
                    result = prefix + result
                    offset += len(prefix)

        # Insert compound emotions at their positions
        for position, emotion in compound_positions:
            adjusted_pos = position + offset

            if engine_type == 'orpheus':
                if emotion in self.ORPHEUS_EMOTION_TAGS:
                    tag = self.ORPHEUS_EMOTION_TAGS[emotion]
                    marker = f' {tag} '
                    result = result[:adjusted_pos] + marker + result[adjusted_pos:]
                    offset += len(marker)
            elif engine_type == 'fishspeech':
                marker_emotion = get_valid_marker(emotion)
                if marker_emotion and marker_emotion in FISH_SPEECH_MARKERS:
                    marker = f' ({marker_emotion}) '
                    result = result[:adjusted_pos] + marker + result[adjusted_pos:]
                    offset += len(marker)

        return result

    def process_segments(
        self,
        segments: List[Tuple[str, str]],
        engine_type: str = 'fishspeech'
    ) -> List[Tuple[str, str, Set[str]]]:
        """
        Process text segments and detect emotions for each.

        This is the main entry point for the narrator mode pipeline.

        Args:
            segments: List of (segment_type, text) tuples from process_text()
                     segment_type is 'narrator', 'character', or 'ambiguous'
            engine_type: 'fishspeech' or 'parler'

        Returns:
            List of (segment_type, text, emotions) tuples
        """
        results = []
        pending_emotions = set()

        for i, (seg_type, text) in enumerate(segments):
            if seg_type == 'narrator':
                # Detect emotions from narrator text
                detected = self.detect_emotions(text)

                if self.apply_to_narrator:
                    # Apply emotions to narrator speech itself
                    results.append((seg_type, text, detected))
                else:
                    # Store emotions to apply to next character/dialogue segment
                    pending_emotions.update(detected)
                    results.append((seg_type, text, set()))

            elif seg_type == 'character':
                if self.apply_to_character:
                    # Combine pending narrator emotions with any in dialogue itself
                    dialogue_emotions = self.detect_emotions(text)
                    all_emotions = pending_emotions | dialogue_emotions

                    # Try LLM if no emotions found and LLM is enabled
                    if not all_emotions and self.use_llm:
                        # Get previous narrator text for context
                        prev_narrator = ''
                        if i > 0 and segments[i-1][0] == 'narrator':
                            prev_narrator = segments[i-1][1]
                        all_emotions = self._llm_infer_emotion(prev_narrator, text)

                    results.append((seg_type, text, all_emotions))
                else:
                    results.append((seg_type, text, set()))

                pending_emotions = set()  # Clear after applying

            else:  # ambiguous
                if self.apply_to_ambiguous:
                    emotions = pending_emotions | self.detect_emotions(text)
                    results.append((seg_type, text, emotions))
                else:
                    results.append((seg_type, text, set()))

                pending_emotions = set()

        return results

    def apply_emotions_to_text(
        self,
        text: str,
        emotions: Set[str],
        engine_type: str = 'fishspeech'
    ) -> str:
        """
        Apply detected emotions to text for a specific TTS engine.

        Args:
            text: The text to modify
            emotions: Set of detected emotions
            engine_type: 'fishspeech', 'orpheus', or 'parler'

        Returns:
            Modified text with emotion markers (Fish Speech/Orpheus) or original text (Parler)
        """
        # Use compound emotions if detection_mode is set to 'compound'
        if self.detection_mode == 'compound':
            return self.apply_compound_emotions(text, engine_type)

        # Basic emotion mode - apply single emotion to entire text
        if engine_type == 'fishspeech':
            return self.apply_fish_speech_markers(text, emotions)
        elif engine_type == 'orpheus':
            return self.apply_orpheus_markers(text, emotions)
        # For Parler, emotions are applied to voice description, not text
        return text


def create_detector(config=None) -> NarrativeEmotionDetector:
    """Factory function to create a detector instance."""
    return NarrativeEmotionDetector(config)

"""
Narrative Emotion Detection Patterns

This module defines regex patterns for detecting emotions and speech styles
from narrative text. Patterns are organized into categories and can be
easily extended by users.

Usage:
    from narrative_patterns import SPEECH_VERBS, EMOTION_ADVERBS, CONTEXT_PHRASES
"""

import re

# =============================================================================
# SPEECH VERBS - How something was said (delivery style)
# =============================================================================
# These patterns detect the manner of speaking and map to delivery-style markers

SPEECH_VERBS = {
    # Quiet/intimate delivery
    r'\b(whispered?|whispers?|whispering)\b': 'whispering',
    r'\b(murmured?|murmurs?|murmuring)\b': 'whispering',
    r'\b(muttered?|mutters?|muttering)\b': 'whispering',
    r'\b(breathed?|breathes?|breathing)\b': 'whispering',  # "she breathed"

    # Loud delivery
    r'\b(shouted?|shouts?|shouting)\b': 'shouting',
    r'\b(yelled?|yells?|yelling)\b': 'shouting',
    r'\b(screamed?|screams?|screaming)\b': 'screaming',
    r'\b(bellowed?|bellows?|bellowing)\b': 'shouting',
    r'\b(roared?|roars?|roaring)\b': 'shouting',
    r'\b(cried out|cries out|crying out)\b': 'shouting',

    # Emotional vocalizations
    r'\b(laughed?|laughs?|laughing)\b': 'laughing',
    r'\b(chuckled?|chuckles?|chuckling)\b': 'laughing',
    r'\b(giggled?|giggles?|giggling)\b': 'laughing',
    r'\b(snickered?|snickers?|snickering)\b': 'laughing',

    r'\b(sobbed?|sobs?|sobbing)\b': 'crying loudly',
    r'\b(wept|weeps?|weeping)\b': 'crying loudly',
    r'\b(cried?|cries?|crying)\b': 'sobbing',  # Less intense than sobbing
    r'\b(sniffled?|sniffles?|sniffling)\b': 'sobbing',

    r'\b(sighed?|sighs?|sighing)\b': 'sighing',
    r'\b(groaned?|groans?|groaning)\b': 'groaning',
    r'\b(panted?|pants?|panting)\b': 'panting',

    # Angry delivery
    r'\b(growled?|growls?|growling)\b': 'angry',
    r'\b(snarled?|snarls?|snarling)\b': 'angry',
    r'\b(snapped?|snaps?|snapping)\b': 'angry',
    r'\b(hissed?|hisses?|hissing)\b': 'angry',
    r'\b(spat|spits?|spitting)\b': 'angry',  # "she spat"

    # Nervous/uncertain delivery
    r'\b(stammered?|stammers?|stammering)\b': 'nervous',
    r'\b(stuttered?|stutters?|stuttering)\b': 'nervous',
    r'\b(faltered?|falters?|faltering)\b': 'nervous',
    r'\b(hesitated?|hesitates?|hesitating)\b': 'hesitating',

    # Pleading
    r'\b(pleaded?|pleads?|pleading)\b': 'nervous',
    r'\b(begged?|begs?|begging)\b': 'nervous',
    r'\b(implored?|implores?|imploring)\b': 'nervous',

    # Surprised
    r'\b(gasped?|gasps?|gasping)\b': 'surprised',
    r'\b(exclaimed?|exclaims?|exclaiming)\b': 'excited',
}

# =============================================================================
# EMOTION ADVERBS - How the speaker felt (emotional tone)
# =============================================================================
# These patterns detect adverbs that describe the emotional state

EMOTION_ADVERBS = {
    # Anger spectrum
    r'\b(angrily|in anger)\b': 'angry',
    r'\b(furiously|with fury)\b': 'furious',
    r'\b(irritably|with irritation)\b': 'angry',
    r'\b(bitterly|with bitterness)\b': 'angry',

    # Sadness spectrum
    r'\b(sadly|with sadness)\b': 'sad',
    r'\b(mournfully|sorrowfully)\b': 'sad',
    r'\b(gloomily|despondently)\b': 'depressed',
    r'\b(wistfully|longingly)\b': 'sad',

    # Joy/excitement spectrum
    r'\b(happily|with happiness)\b': 'joyful',
    r'\b(joyfully|with joy)\b': 'joyful',
    r'\b(excitedly|with excitement)\b': 'excited',
    r'\b(enthusiastically)\b': 'excited',
    r'\b(eagerly|with eagerness)\b': 'excited',
    r'\b(cheerfully|cheerily)\b': 'joyful',

    # Fear/anxiety spectrum
    r'\b(nervously|with nerves)\b': 'nervous',
    r'\b(anxiously|with anxiety)\b': 'anxious',
    r'\b(fearfully|with fear)\b': 'scared',
    r'\b(timidly|shyly)\b': 'nervous',
    r'\b(hesitantly|uncertainly)\b': 'hesitating',

    # Softness/gentleness
    r'\b(softly|gently)\b': 'soft tone',
    r'\b(tenderly|lovingly)\b': 'soft tone',
    r'\b(quietly|in a low voice)\b': 'soft tone',
    r'\b(sweetly|kindly)\b': 'soft tone',

    # Coldness/detachment
    r'\b(coldly|icily)\b': 'serious',
    r'\b(flatly|emotionlessly)\b': 'indifferent',
    r'\b(curtly|shortly)\b': 'serious',
    r'\b(dismissively)\b': 'disdainful',

    # Sarcasm/mockery
    r'\b(sarcastically|with sarcasm)\b': 'sarcastic',
    r'\b(mockingly|tauntingly)\b': 'sarcastic',
    r'\b(snidely|sneeringly)\b': 'sneering',
    r'\b(dryly|wryly)\b': 'sarcastic',

    # Confidence/authority
    r'\b(confidently|with confidence)\b': 'confident',
    r'\b(firmly|sternly)\b': 'serious',
    r'\b(proudly|with pride)\b': 'proud',
    r'\b(defiantly)\b': 'angry',

    # Speed/urgency
    r'\b(hurriedly|hastily)\b': 'in a hurry tone',
    r'\b(urgently|desperately)\b': 'in a hurry tone',
    r'\b(breathlessly)\b': 'panting',

    # Surprise
    r'\b(surprisingly|with surprise)\b': 'surprised',
    r'\b(incredulously|in disbelief)\b': 'astonished',

    # Embarrassment
    r'\b(sheepishly|embarrassedly)\b': 'embarrassed',
    r'\b(awkwardly|uncomfortably)\b': 'awkward',
}

# =============================================================================
# CONTEXT PHRASES - Descriptive context clues
# =============================================================================
# These patterns detect descriptive phrases that imply emotion

CONTEXT_PHRASES = {
    # Physical indicators of emotion
    r'\bwith a smile\b': 'joyful',
    r'\bwith a grin\b': 'joyful',
    r'\bthrough (her |his )?tears\b': 'sad',
    r'\bvoice (trembling|shaking)\b': 'nervous',
    r'\bhands (trembling|shaking)\b': 'nervous',
    r'\bwith (a )?trembling (voice|lip)\b': 'nervous',
    r'\bin a rush\b': 'in a hurry tone',
    r'\bin a hurry\b': 'in a hurry tone',
    r'\bunder (her |his )?breath\b': 'whispering',
    r'\bbarely audible\b': 'whispering',
    r'\bvoice (rising|raised)\b': 'shouting',
    r'\bvoice (dropping|lowered)\b': 'whispering',
    r'\beyes (wide|widening)\b': 'surprised',
    r'\bwith (a )?sigh\b': 'sighing',
    r'\bwith (a )?laugh\b': 'laughing',
    r'\bwith (a )?sob\b': 'sobbing',
    r'\bfighting back tears\b': 'sad',
    r'\bholding back (a )?laugh\b': 'amused',
    r'\bface (red|flushed|blushing)\b': 'embarrassed',
    r'\brolling (her |his )?eyes\b': 'sarcastic',
}

# =============================================================================
# FISH SPEECH MARKER MAPPING
# =============================================================================
# Map detected emotions to valid Fish Speech markers
# This ensures we only use markers that Fish Speech actually supports

FISH_SPEECH_MARKERS = {
    # Direct mappings (emotion name = marker name)
    'angry', 'sad', 'excited', 'surprised', 'satisfied', 'unhappy',
    'anxious', 'hysterical', 'delighted', 'scared', 'worried', 'indifferent',
    'upset', 'impatient', 'nervous', 'guilty', 'scornful', 'frustrated',
    'depressed', 'panicked', 'furious', 'empathetic', 'embarrassed', 'reluctant',
    'disgusted', 'keen', 'moved', 'proud', 'relaxed', 'grateful', 'confident',
    'interested', 'curious', 'confused', 'joyful', 'disapproving', 'negative',
    'denying', 'astonished', 'serious', 'sarcastic', 'conciliative', 'comforting',
    'sincere', 'sneering', 'hesitating', 'yielding', 'painful', 'awkward', 'amused',
    # Tone markers
    'whispering', 'shouting', 'screaming', 'soft tone', 'in a hurry tone',
    # Effect markers
    'laughing', 'chuckling', 'sobbing', 'crying loudly', 'sighing', 'panting', 'groaning',
}

# Fallback mappings for emotions that don't have direct Fish Speech equivalents
EMOTION_FALLBACK = {
    'disdainful': 'scornful',
}


def compile_patterns():
    """Compile all regex patterns for efficient matching."""
    compiled = {
        'speech_verbs': [(re.compile(p, re.IGNORECASE), e) for p, e in SPEECH_VERBS.items()],
        'emotion_adverbs': [(re.compile(p, re.IGNORECASE), e) for p, e in EMOTION_ADVERBS.items()],
        'context_phrases': [(re.compile(p, re.IGNORECASE), e) for p, e in CONTEXT_PHRASES.items()],
    }
    return compiled


# Pre-compiled patterns for performance
COMPILED_PATTERNS = compile_patterns()


def get_valid_marker(emotion):
    """Ensure the emotion maps to a valid Fish Speech marker."""
    if emotion in FISH_SPEECH_MARKERS:
        return emotion
    if emotion in EMOTION_FALLBACK:
        return EMOTION_FALLBACK[emotion]
    # Default fallback
    return None

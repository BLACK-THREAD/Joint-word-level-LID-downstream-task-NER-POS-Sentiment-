from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def romanized_hindi_to_devanagari(text):
    """
    Convert Romanized Hindi to Devanagari.
    Uses ITRANS transliteration scheme.
    
    Args:
        text (str): Romanized Hindi text
        
    Returns:
        str: Devanagari text
    """
    return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)


# Manual mapping for better Hinglish conversion
hindi_word_map = {
    'app': 'आप',
    'aap': 'आप',
    'kaisa': 'कैसा',
    'kaise': 'कैसे',
    'ho': 'हो',
    'main': 'मैं',
    'theek': 'ठीक',
    'hoon': 'हूं',
    'kya': 'क्या',
    'haal': 'हाल',
    'hai': 'है',
    'kal': 'कल',
    'milte': 'मिलते',
    'aaj': 'आज',
    'ka': 'का',
    'plan': 'प्लान',
    'namaste': 'नमस्ते',
    'bhai': 'भाई',
    'dost': 'दोस्त'
}

# Common English words to keep unchanged
english_words = {'bro', 'dude', 'hello', 'hi', 'bye', 'ok', 'okay', 'yes', 'no', 'the', 'a', 'an', 'how', 'are', 'you'}


def smart_hinglish_converter(text):
    """
    Convert Hinglish to mixed Devanagari-English.
    Hindi words → Devanagari
    English words → Keep as is
    
    Args:
        text (str): Input Hinglish text
        
    Returns:
        str: Mixed script output
    """
    words = text.split()
    result = []
    
    for word in words:
        # Remove punctuation for checking
        clean_word = word.strip('.,!?')
        punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ''
        
        # Check if it's a known English word
        if clean_word.lower() in english_words:
            result.append(word)
        # Check if it's in our Hindi dictionary
        elif clean_word.lower() in hindi_word_map:
            result.append(hindi_word_map[clean_word.lower()] + punctuation)
        else:
            # Try transliteration
            try:
                transliterated = transliterate(clean_word, sanscript.ITRANS, sanscript.DEVANAGARI)
                result.append(transliterated + punctuation)
            except:
                result.append(word)
    
    return ' '.join(result)


# Test examples
print("=== Smart Hinglish → Devanagari Converter ===\n")

test_cases = [
    "app kaisa ho bro kal milte?",
    "main theek hoon how are you",
    "kya haal hai dude",
    "aaj ka plan kya hai",
    "hello namaste kaise ho"
]

for text in test_cases:
    result = smart_hinglish_converter(text)
    print(f"Input:  {text}")
    print(f"Output: {result}")
    print()

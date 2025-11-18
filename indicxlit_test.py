from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import fasttext
import warnings

warnings.filterwarnings('ignore')

print("Loading FastText LID model...")
lid_model = fasttext.load_model('lid.176.bin')
print("Model loaded!\n")


def transliterate_hinglish(text):
    """
    Convert Romanized Hindi to Devanagari while keeping English words unchanged.
    
    Args:
        text (str): Input Hinglish text (e.g., "app kaisa ho bro")
        
    Returns:
        str: Transliterated text (e.g., "आप कैसा हो bro")
    """
    words = text.split()
    result = []
    
    for word in words:
        # Check if word is likely Hindi (romanized) or English
        # Simple heuristic: try to detect language
        labels, scores = lid_model.predict(word.lower())
        lang = labels[0].replace("__label__", "")
        
        # If detected as English with high confidence, keep it
        if lang == 'en' and scores[0] > 0.7:
            result.append(word)
        else:
            # Try to transliterate from Roman to Devanagari
            try:
                transliterated = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
                result.append(transliterated)
            except:
                result.append(word)
    
    return ' '.join(result)


# Test examples
test_cases = [
    "app kaisa ho bro kal milte?",
    "main theek hoon how are you",
    "kya haal hai dude",
    "aaj ka plan kya hai",
    "hello namaste kaise ho"
]

print("=== Hinglish → Devanagari Transliteration ===\n")
for text in test_cases:
    result = transliterate_hinglish(text)
    print(f"Input:  {text}")
    print(f"Output: {result}")
    print()

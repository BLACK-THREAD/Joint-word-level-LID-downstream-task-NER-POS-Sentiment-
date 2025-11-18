import fasttext
import warnings
warnings.filterwarnings('ignore')

# Load the model
model = fasttext.load_model('lid.176.bin')


def get_word_lids(text):
    """
    Get language IDs for each word in the input text.
    
    Args:
        text (str): Input text with multiple words
        
    Returns:
        list: List of language IDs for each word
    """
    words = text.split()
    lids = []
    
    for word in words:
        labels, scores = model.predict(word)
        # Remove '__label__' prefix and get just the language code
        lid = labels[0].replace("__label__", "")
        lids.append(lid)
    
    return lids


# Test Hinglish vs proper Hindi
print("=== Romanized Hindi (Hinglish) ===")
hinglish_texts = [
    "app kaisa ho",
    "main theek hoon", 
    "kya haal hai"
]

for text in hinglish_texts:
    labels, scores = model.predict(text)
    print(f"Text: {text}")
    print(f"Detected: {labels[0].replace('__label__', '')} (confidence: {scores[0]:.4f})")
    print()

print("=== Proper Hindi (Devanagari) ===")
hindi_texts = [
    "आप कैसे हो",
    "मैं ठीक हूं",
    "क्या हाल है"
]

for text in hindi_texts:
    labels, scores = model.predict(text)
    print(f"Text: {text}")
    print(f"Detected: {labels[0].replace('__label__', '')} (confidence: {scores[0]:.4f})")
    print()

print("=== Code-Mixed (Hinglish) ===")
mixed = "app kaisa ho hello"
result = get_word_lids(mixed)
print(f"Input: {mixed}")
print(f"Word-by-word LIDs: {result}")
print("\nNote: FastText doesn't recognize Romanized Hindi or code-mixed languages.")
print("It only detects Hindi when written in Devanagari script.")

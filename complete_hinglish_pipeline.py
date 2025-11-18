import fasttext
import warnings
from transformers import pipeline

warnings.filterwarnings('ignore')

print("Loading all models...")
print("=" * 70)

# Load FastText LID
print("1. FastText LID...")
lid_model = fasttext.load_model('lid.176.bin')

# Load NLP models
print("2. Sentiment Analysis...")
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", framework="pt")

print("3. NER...")
ner_analyzer = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple", framework="pt")

print("4. POS Tagging...")
pos_analyzer = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos", framework="pt")

print("=" * 70)
print("✓ All models loaded!\n")


# Hindi dictionaries (from previous code)
hindi_to_english = {
    'मैं': 'I', 'तुम': 'you', 'आप': 'you', 'हम': 'we', 'वो': 'that',
    'है': 'is', 'हैं': 'are', 'हो': 'are', 'हूं': 'am',
    'क्या': 'what', 'क्यों': 'why', 'कब': 'when', 'कहां': 'where',
    'कौन': 'who', 'कैसे': 'how', 'कैसा': 'how',
    'ठीक': 'fine', 'अच्छा': 'good', 'हाल': 'condition',
    'कल': 'tomorrow', 'आज': 'today', 'मेरा': 'my',
    'का': 'of', 'को': 'to', 'से': 'from', 'में': 'in',
    'नमस्ते': 'hello', 'नाम': 'name', 'यार': 'friend',
    'मिलते': 'meet', 'प्लान': 'plan',
}

hindi_dict = {
    'main': 'मैं', 'mein': 'मैं', 'app': 'आप', 'aap': 'आप',
    'tum': 'तुम', 'hum': 'हम', 'woh': 'वो', 'wo': 'वो',
    'hai': 'है', 'hain': 'हैं', 'ho': 'हो', 'hoon': 'हूं', 'hun': 'हूं',
    'kya': 'क्या', 'kyu': 'क्यों', 'kab': 'कब', 'kahan': 'कहां',
    'kaun': 'कौन', 'kaise': 'कैसे', 'kaisa': 'कैसा',
    'theek': 'ठीक', 'thik': 'ठीक', 'accha': 'अच्छा', 'acha': 'अच्छा',
    'haal': 'हाल', 'kal': 'कल', 'aaj': 'आज',
    'mera': 'मेरा', 'meri': 'मेरी', 'mere': 'मेरे',
    'ka': 'का', 'ki': 'की', 'ke': 'के',
    'ko': 'को', 'se': 'से', 'me': 'में', 'mein': 'में',
    'namaste': 'नमस्ते', 'naam': 'नाम', 'yaar': 'यार',
    'milte': 'मिलते', 'plan': 'प्लान',
}

english_words = {
    'bro', 'dude', 'hello', 'hi', 'hey', 'bye', 'okay', 'ok',
    'how', 'are', 'you', 'i', 'am', 'is', 'was', 'were',
    'good', 'bad', 'nice', 'great', 'awesome', 'cool',
    'today', 'tomorrow', 'my', 'your',
}


def detect_word_language(word):
    clean_word = word.lower().strip('.,!?;:')
    if clean_word in english_words:
        return 'en'
    if clean_word in hindi_dict:
        return 'hi'
    try:
        labels, scores = lid_model.predict(clean_word)
        lang = labels[0].replace("__label__", "")
        if lang == 'en' and scores[0] > 0.6:
            return 'en'
        else:
            return 'hi'
    except:
        return 'hi'


def complete_hinglish_analysis(text):
    """
    Complete Hinglish NLP pipeline:
    - Convert to mixed script
    - LID detection
    - English translation
    - Sentiment, NER, POS analysis
    """
    words = text.split()
    result_words = []
    lids = []
    english_translation = []
    
    # Convert and detect
    for word in words:
        clean_word = word.strip('.,!?;:')
        punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ''
        
        lang = detect_word_language(clean_word)
        lids.append(lang)
        
        if lang == 'en':
            result_words.append(word)
            english_translation.append(word)
        else:
            hindi_word = hindi_dict.get(clean_word.lower(), clean_word)
            result_words.append(hindi_word + punctuation)
            english_word = hindi_to_english.get(hindi_word, clean_word)
            english_translation.append(english_word + punctuation)
    
    output = ' '.join(result_words)
    ml = ' '.join(english_translation)
    hindi_count = lids.count('hi')
    english_count = lids.count('en')
    mli = 'Hindi' if hindi_count >= english_count else 'English'
    
    # NLP Analysis
    try:
        sentiment_result = sentiment_analyzer(text)[0]
        sentiment = f"{sentiment_result['label']} ({sentiment_result['score']:.4f})"
    except:
        sentiment = "N/A"
    
    try:
        ner_results = ner_analyzer(text)
        entities = [f"{e['word']} [{e['entity_group']}]" for e in ner_results]
        ner = entities if entities else ["None"]
    except:
        ner = ["Error"]
    
    try:
        pos_results = pos_analyzer(text)
        pos_tags = [f"{p['word']}→{p['entity']}" for p in pos_results[:8]]
        pos = pos_tags if pos_tags else ["None"]
    except:
        pos = ["Error"]
    
    return {
        'input': text,
        'output': output,
        'lids': lids,
        'ml': ml,
        'mli': mli,
        'sentiment': sentiment,
        'ner': ner,
        'pos': pos
    }


# Test
print("=" * 70)
print("COMPLETE HINGLISH NLP PIPELINE")
print("=" * 70)
print()

test_cases = [
    "app kaisa ho bro kal milte?",
    "main theek hoon how are you",
    "Rahul Delhi ja raha hai tomorrow",
]

for text in test_cases:
    result = complete_hinglish_analysis(text)
    
    print(f"📝 Input:      {result['input']}")
    print(f"📄 Output:     {result['output']}")
    print(f"🔤 LIDs:       {result['lids']}")
    print(f"🌐 ML:         {result['ml']}")
    print(f"🗣️  MLI:        {result['mli']}")
    print(f"😊 Sentiment:  {result['sentiment']}")
    print(f"🏷️  NER:        {', '.join(result['ner'])}")
    print(f"📌 POS:        {', '.join(result['pos'])}")
    print()
    print("-" * 70)
    print()

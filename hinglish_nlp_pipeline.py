import fasttext
import warnings
from transformers import pipeline
import torch

warnings.filterwarnings('ignore')

print("Loading models (this may take a few minutes on first run)...")
print("=" * 60)

# Load FastText LID
print("1. Loading FastText LID model...")
lid_model = fasttext.load_model('lid.176.bin')
print("   ✓ FastText LID loaded")

# Load Sentiment Analysis model (multilingual)
print("2. Loading Sentiment Analysis model...")
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", framework="pt")
print("   ✓ Sentiment model loaded")

# Load NER model (multilingual)
print("3. Loading NER model...")
ner_analyzer = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple", framework="pt")
print("   ✓ NER model loaded")

# Load POS model
print("4. Loading POS Tagging model...")
pos_analyzer = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos", framework="pt")
print("   ✓ POS model loaded")

print("=" * 60)
print("All models loaded successfully!\n")


def analyze_hinglish_text(text):
    """
    Complete NLP analysis for Hinglish text.
    
    Returns:
        dict with sentiment, NER, POS analysis
    """
    print(f"Analyzing: {text}")
    print("-" * 60)
    
    # 1. Sentiment Analysis
    try:
        sentiment_result = sentiment_analyzer(text)[0]
        sentiment = {
            'label': sentiment_result['label'],
            'score': round(sentiment_result['score'], 4)
        }
    except Exception as e:
        sentiment = {'label': 'Error', 'score': 0.0}
    
    # 2. Named Entity Recognition
    try:
        ner_results = ner_analyzer(text)
        entities = []
        for entity in ner_results:
            entities.append({
                'text': entity['word'],
                'type': entity['entity_group'],
                'score': round(entity['score'], 4)
            })
        ner = entities if entities else [{'text': 'None', 'type': 'N/A', 'score': 0.0}]
    except Exception as e:
        ner = [{'text': 'Error', 'type': 'N/A', 'score': 0.0}]
    
    # 3. POS Tagging
    try:
        pos_results = pos_analyzer(text)
        pos_tags = []
        for token in pos_results:
            pos_tags.append({
                'word': token['word'],
                'pos': token['entity'],
                'score': round(token['score'], 4)
            })
        pos = pos_tags if pos_tags else [{'word': 'None', 'pos': 'N/A', 'score': 0.0}]
    except Exception as e:
        pos = [{'word': 'Error', 'pos': 'N/A', 'score': 0.0}]
    
    return {
        'sentiment': sentiment,
        'ner': ner,
        'pos': pos
    }


# Test examples
test_cases = [
    "app kaisa ho bro kal milte?",
    "main theek hoon how are you",
    "Rahul is going to Delhi tomorrow",
    "I love this movie bahut accha hai"
]

print("=" * 60)
print("HINGLISH NLP ANALYSIS")
print("=" * 60)
print()

for text in test_cases:
    result = analyze_hinglish_text(text)
    
    print(f"📝 Input: {text}")
    print(f"😊 Sentiment: {result['sentiment']['label']} (confidence: {result['sentiment']['score']})")
    print(f"🏷️  NER Entities:")
    for entity in result['ner']:
        print(f"   - {entity['text']} [{entity['type']}] (confidence: {entity['score']})")
    print(f"📌 POS Tags:")
    for pos in result['pos'][:5]:  # Show first 5 tokens
        print(f"   - {pos['word']} → {pos['pos']}")
    print()
    print("-" * 60)
    print()

import fasttext
import warnings
from langdetect import detect, LangDetectException

warnings.filterwarnings('ignore')

print("Loading FastText LID model...")
lid_model = fasttext.load_model('lid.176.bin')
print("Model loaded!\n")

# Hindi to English translation dictionary
hindi_to_english = {
    'मैं': 'I', 'तू': 'you', 'तुम': 'you', 'आप': 'you',
    'हम': 'we', 'वो': 'that', 'यह': 'this', 'ये': 'these',
    'है': 'is', 'हैं': 'are', 'हो': 'are', 'हूं': 'am',
    'था': 'was', 'थी': 'was', 'थे': 'were', 'थीं': 'were',
    'कर': 'do', 'करो': 'do', 'करना': 'to do', 'किया': 'did',
    'जा': 'go', 'जाओ': 'go', 'जाना': 'to go', 'गया': 'went',
    'आ': 'come', 'आओ': 'come', 'आना': 'to come', 'आया': 'came',
    'दे': 'give', 'दो': 'give', 'देना': 'to give', 'दिया': 'gave',
    'ले': 'take', 'लो': 'take', 'लेना': 'to take', 'लिया': 'took',
    'मिलते': 'meet', 'मिलता': 'meets', 'मिला': 'met', 'मिले': 'met',
    'क्या': 'what', 'क्यों': 'why', 'कब': 'when', 'कहां': 'where',
    'कौन': 'who', 'कैसे': 'how', 'कैसा': 'how', 'कैसी': 'how',
    'कितना': 'how much', 'कितने': 'how many', 'कितनी': 'how many',
    'ठीक': 'fine', 'अच्छा': 'good', 'बुरा': 'bad',
    'हाल': 'condition', 'कल': 'tomorrow/yesterday', 'आज': 'today', 'परसों': 'day after/before',
    'अभी': 'now', 'अब': 'now',
    'यहां': 'here', 'वहां': 'there',
    'मेरा': 'my', 'मेरी': 'my', 'मेरे': 'my',
    'तेरा': 'your', 'तेरी': 'your', 'तेरे': 'your',
    'तुम्हारा': 'your', 'तुम्हारी': 'your', 'तुम्हारे': 'your',
    'उसका': 'his/her', 'उसकी': 'his/her', 'उसके': 'his/her',
    'का': 'of', 'की': 'of', 'के': 'of',
    'को': 'to', 'से': 'from', 'में': 'in', 'पर': 'on', 'पे': 'on',
    'तक': 'until', 'के लिए': 'for',
    'नमस्ते': 'hello', 'नमस्कार': 'hello',
    'शुक्रिया': 'thank you', 'धन्यवाद': 'thank you',
    'घर': 'home', 'काम': 'work',
    'दिन': 'day', 'रात': 'night', 'सुबह': 'morning', 'शाम': 'evening',
    'दोस्त': 'friend', 'भाई': 'brother', 'बहन': 'sister',
    'मां': 'mother', 'बाप': 'father', 'पापा': 'dad', 'मम्मी': 'mom',
    'प्लान': 'plan', 'टाइम': 'time',
    'नाम': 'name', 'याद': 'remember', 'यार': 'friend',
    'नहीं': 'no', 'ना': 'no', 'हां': 'yes',
    'भी': 'also', 'और': 'and', 'या': 'or',
    'लेकिन': 'but', 'बहुत': 'very', 'थोड़ा': 'little', 'ज्यादा': 'more',
}

# Comprehensive Hindi word dictionary (Romanized → Devanagari)
hindi_dict = {
    # Pronouns
    'main': 'मैं', 'mein': 'मैं', 'mai': 'मैं',
    'tu': 'तू', 'tum': 'तुम', 'aap': 'आप', 'app': 'आप',
    'hum': 'हम', 'woh': 'वो', 'wo': 'वो', 'yeh': 'यह', 'ye': 'ये',
    
    # Verbs
    'hai': 'है', 'hain': 'हैं', 'ho': 'हो', 'hoon': 'हूं', 'hun': 'हूं',
    'tha': 'था', 'thi': 'थी', 'the': 'थे', 'thi': 'थीं',
    'kar': 'कर', 'karo': 'करो', 'karna': 'करना', 'kiya': 'किया',
    'ja': 'जा', 'jao': 'जाओ', 'jana': 'जाना', 'gaya': 'गया',
    'aa': 'आ', 'aao': 'आओ', 'aana': 'आना', 'aaya': 'आया',
    'de': 'दे', 'do': 'दो', 'dena': 'देना', 'diya': 'दिया',
    'le': 'ले', 'lo': 'लो', 'lena': 'लेना', 'liya': 'लिया',
    'milte': 'मिलते', 'milta': 'मिलता', 'mila': 'मिला', 'mile': 'मिले',
    
    # Question words
    'kya': 'क्या', 'kyu': 'क्यों', 'kyun': 'क्यों', 'kyon': 'क्यों',
    'kab': 'कब', 'kahan': 'कहां', 'kaha': 'कहां',
    'kaun': 'कौन', 'kon': 'कौन',
    'kaise': 'कैसे', 'kaisa': 'कैसा', 'kaisi': 'कैसी',
    'kitna': 'कितना', 'kitne': 'कितने', 'kitni': 'कितनी',
    
    # Common words
    'theek': 'ठीक', 'thik': 'ठीक', 'accha': 'अच्छा', 'acha': 'अच्छा',
    'bura': 'बुरा', 'achha': 'अच्छा',
    'haal': 'हाल', 'hal': 'हाल',
    'kal': 'कल', 'aaj': 'आज', 'parso': 'परसों',
    'abhi': 'अभी', 'ab': 'अब',
    'yaha': 'यहां', 'yahan': 'यहां', 'waha': 'वहां', 'wahan': 'वहां',
    
    # Possessives
    'mera': 'मेरा', 'meri': 'मेरी', 'mere': 'मेरे',
    'tera': 'तेरा', 'teri': 'तेरी', 'tere': 'तेरे',
    'tumhara': 'तुम्हारा', 'tumhari': 'तुम्हारी', 'tumhare': 'तुम्हारे',
    'uska': 'उसका', 'uski': 'उसकी', 'uske': 'उसके',
    
    # Postpositions
    'ka': 'का', 'ki': 'की', 'ke': 'के',
    'ko': 'को', 'se': 'से', 'me': 'में', 'mein': 'में',
    'par': 'पर', 'pe': 'पे',
    'tak': 'तक', 'ke liye': 'के लिए',
    
    # Greetings
    'namaste': 'नमस्ते', 'namaskar': 'नमस्कार',
    'shukriya': 'शुक्रिया', 'dhanyavaad': 'धन्यवाद',
    
    # Common nouns
    'ghar': 'घर', 'kaam': 'काम', 'kam': 'काम',
    'din': 'दिन', 'raat': 'रात', 'subah': 'सुबह', 'shaam': 'शाम',
    'dost': 'दोस्त', 'bhai': 'भाई', 'behen': 'बहन',
    'maa': 'मां', 'baap': 'बाप', 'papa': 'पापा', 'mummy': 'मम्मी',
    'plan': 'प्लान', 'time': 'टाइम',
    'naam': 'नाम', 'yaad': 'याद', 'yaar': 'यार',
    
    # Others
    'nahi': 'नहीं', 'nahin': 'नहीं', 'na': 'ना',
    'haan': 'हां', 'ha': 'हां',
    'bhi': 'भी', 'bhe': 'भी',
    'aur': 'और', 'or': 'या',
    'lekin': 'लेकिन', 'par': 'पर',
    'bahut': 'बहुत', 'bohot': 'बहुत',
    'thoda': 'थोड़ा', 'jyada': 'ज्यादा', 'zyada': 'ज्यादा',
}

# Common English words to keep unchanged
english_words = {
    'bro', 'dude', 'hello', 'hi', 'hey', 'bye', 'okay', 'ok', 'yes', 'no',
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'how', 'are', 'you', 'i', 'am', 'is', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could',
    'what', 'when', 'where', 'who', 'why', 'which',
    'good', 'bad', 'nice', 'great', 'awesome', 'cool', 'man', 'guy',
    'today', 'tomorrow', 'yesterday', 'now', 'then', 'here', 'there',
    'my', 'your', 'his', 'her', 'our', 'their', 'its',
    'this', 'that', 'these', 'those', 'some', 'any', 'all', 'many', 'much',
}


def detect_word_language(word):
    """
    Detect if a word is English or Hindi (romanized).
    Returns: 'en' for English, 'hi' for Hindi
    """
    clean_word = word.lower().strip('.,!?;:')
    
    # Check if it's a known English word
    if clean_word in english_words:
        return 'en'
    
    # Check if it's a known Hindi word
    if clean_word in hindi_dict:
        return 'hi'
    
    # Use FastText LID for unknown words
    try:
        labels, scores = lid_model.predict(clean_word)
        lang = labels[0].replace("__label__", "")
        
        # If detected as English with high confidence, it's English
        if lang == 'en' and scores[0] > 0.6:
            return 'en'
        # Otherwise assume Hindi (for Hinglish context)
        else:
            return 'hi'
    except:
        return 'hi'  # Default to Hindi for unknown


def convert_hinglish_to_mixed(text):
    """
    Convert Hinglish to mixed Devanagari-English.
    
    Input: "app kaisa ho bro kal milte?"
    Output: "आप कैसा हो bro कल मिलते?"
    
    Args:
        text (str): Input Hinglish text
        
    Returns:
        dict: {
            'output': Mixed script output,
            'lids': List of LIDs,
            'ml': Matrix Language (English translation),
            'mli': Matrix Language Index (main language)
        }
    """
    words = text.split()
    result = []
    lids = []
    english_translation = []
    
    for word in words:
        # Preserve punctuation
        clean_word = word.strip('.,!?;:')
        punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ''
        
        # Detect language
        lang = detect_word_language(clean_word)
        lids.append(lang)
        
        if lang == 'en':
            # Keep English words unchanged
            result.append(word)
            english_translation.append(word)
        else:
            # Convert Hindi words to Devanagari
            hindi_word = hindi_dict.get(clean_word.lower(), clean_word)
            result.append(hindi_word + punctuation)
            
            # Translate to English
            english_word = hindi_to_english.get(hindi_word, clean_word)
            english_translation.append(english_word + punctuation)
    
    # Calculate MLI (Matrix Language Index) - main language of sentence
    hindi_count = lids.count('hi')
    english_count = lids.count('en')
    mli = 'Hindi' if hindi_count >= english_count else 'English'
    
    return {
        'output': ' '.join(result),
        'lids': lids,
        'ml': ' '.join(english_translation),
        'mli': mli
    }


# Test examples
print("=== Correct Hinglish → Mixed Script Converter ===\n")

test_cases = [
    "app kaisa ho bro kal milte?",
    "main theek hoon how are you",
    "kya haal hai dude",
    "aaj ka plan kya hai",
    "hello namaste kaise ho",
    "tum kahan ho yaar",
    "mera naam kya hai bro",
]

for text in test_cases:
    result = convert_hinglish_to_mixed(text)
    print(f"Input:  {text}")
    print(f"Output: {result['output']}")
    print(f"LIDs:   {result['lids']}")
    print(f"ML:     {result['ml']}")
    print(f"MLI:    {result['mli']}")
    print()

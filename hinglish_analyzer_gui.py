import tkinter as tk
from tkinter import ttk, scrolledtext
import fasttext
import warnings
from transformers import pipeline
import threading

warnings.filterwarnings('ignore')

# Load models
print("Loading models...")
lid_model = fasttext.load_model('lid.176.bin')
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", framework="pt")
ner_analyzer = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple", framework="pt")
pos_analyzer = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos", framework="pt")
print("✓ Models loaded!")

# Dictionaries
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
        return 'en' if lang == 'en' and scores[0] > 0.6 else 'hi'
    except:
        return 'hi'


def analyze_text(text):
    words = text.split()
    result_words = []
    lids = []
    english_translation = []
    
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
    mli = 'Hindi' if lids.count('hi') >= lids.count('en') else 'English'
    
    try:
        sentiment_result = sentiment_analyzer(text)[0]
        sentiment = f"{sentiment_result['label']} ({sentiment_result['score']:.4f})"
    except:
        sentiment = "N/A"
    
    try:
        ner_results = ner_analyzer(text)
        entities = [(e['word'], e['entity_group']) for e in ner_results]
    except:
        entities = []
    
    try:
        pos_results = pos_analyzer(text)
        pos_tags = [(p['word'], p['entity']) for p in pos_results]
    except:
        pos_tags = []
    
    return {
        'output': output,
        'lids': lids,
        'ml': ml,
        'mli': mli,
        'sentiment': sentiment,
        'entities': entities,
        'pos_tags': pos_tags
    }



class HinglishAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hinglish Analyzer")
        self.root.geometry("1000x800")
        self.root.configure(bg='#1e1e1e')
        
        # Colors
        self.bg_dark = '#1e1e1e'
        self.bg_medium = '#2d2d2d'
        self.bg_light = '#3d3d3d'
        self.text_color = '#e0e0e0'
        self.accent = '#4a9eff'
        
        # POS colors
        self.pos_colors = {
            'NOUN': '#ff6b6b',
            'VERB': '#4ecdc4',
            'ADJ': '#a78bfa',
            'ADV': '#fbbf24',
            'PROPN': '#34d399',
            'PRON': '#f472b6',
            'DET': '#60a5fa',
            'ADP': '#fb923c',
            'INTJ': '#c084fc',
        }
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="🌐 Hinglish Text Analyzer", 
                        font=('Arial', 24, 'bold'), bg=self.bg_dark, fg=self.accent)
        title.pack(pady=20)
        
        # Input frame
        input_frame = tk.Frame(self.root, bg=self.bg_dark)
        input_frame.pack(pady=10, padx=20, fill='x')
        
        input_label = tk.Label(input_frame, text="📝 Input Text:", 
                              font=('Arial', 12, 'bold'), bg=self.bg_dark, fg=self.text_color)
        input_label.pack(anchor='w')
        
        self.input_text = tk.Text(input_frame, height=3, font=('Arial', 12),
                                 bg=self.bg_medium, fg=self.text_color,
                                 insertbackground=self.text_color, relief='flat',
                                 padx=10, pady=10)
        self.input_text.pack(fill='x', pady=5)
        
        # Analyze button
        analyze_btn = tk.Button(self.root, text="🔍 Analyze", command=self.analyze,
                               font=('Arial', 14, 'bold'), bg=self.accent, fg='white',
                               relief='flat', padx=30, pady=10, cursor='hand2')
        analyze_btn.pack(pady=10)
        
        # Output frame
        output_frame = tk.Frame(self.root, bg=self.bg_dark)
        output_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        output_label = tk.Label(output_frame, text="📄 Analysis Results:", 
                               font=('Arial', 12, 'bold'), bg=self.bg_dark, fg=self.text_color)
        output_label.pack(anchor='w')
        
        # Output text with tags
        self.output_text = tk.Text(output_frame, font=('Arial', 11),
                                  bg=self.bg_medium, fg=self.text_color,
                                  relief='flat', padx=15, pady=15, wrap='word')
        self.output_text.pack(fill='both', expand=True, pady=5)
        
        # Configure tags for colors
        self.output_text.tag_config('title', foreground=self.accent, font=('Arial', 11, 'bold'))
        self.output_text.tag_config('entity', foreground='#34d399', font=('Arial', 11, 'bold'))
        for pos, color in self.pos_colors.items():
            self.output_text.tag_config(pos, foreground=color, font=('Arial', 11, 'bold'))
        
        # Status
        self.status_label = tk.Label(self.root, text="Ready", 
                                     font=('Arial', 10), bg=self.bg_dark, fg='#888888')
        self.status_label.pack(pady=5)

    
    def analyze(self):
        input_text = self.input_text.get("1.0", "end-1c").strip()
        if not input_text:
            self.status_label.config(text="⚠️ Please enter some text", fg='#ff6b6b')
            return
        
        self.status_label.config(text="🔄 Analyzing...", fg='#fbbf24')
        self.root.update()
        
        # Run analysis in thread
        thread = threading.Thread(target=self.run_analysis, args=(input_text,))
        thread.start()
    
    def run_analysis(self, text):
        try:
            result = analyze_text(text)
            self.root.after(0, self.display_results, text, result)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def show_error(self, error):
        self.status_label.config(text=f"❌ Error: {error}", fg='#ff6b6b')
    
    def display_results(self, input_text, result):
        self.output_text.delete("1.0", "end")
        
        # Input
        self.output_text.insert("end", "📝 Input:\n", 'title')
        self.output_text.insert("end", f"   {input_text}\n\n")
        
        # Output with POS highlighting
        self.output_text.insert("end", "📄 Output (with POS tags):\n", 'title')
        self.output_text.insert("end", "   ")
        
        output_words = result['output'].split()
        pos_dict = {p[0]: p[1] for p in result['pos_tags']}
        entity_dict = {e[0]: e[1] for e in result['entities']}
        
        for word in output_words:
            clean_word = word.strip('.,!?;:')
            
            # Check if entity
            if clean_word in entity_dict:
                self.output_text.insert("end", word + " ", 'entity')
            # Check POS
            elif clean_word in pos_dict:
                pos = pos_dict[clean_word]
                if pos in self.pos_colors:
                    self.output_text.insert("end", word + " ", pos)
                else:
                    self.output_text.insert("end", word + " ")
            else:
                self.output_text.insert("end", word + " ")
        
        self.output_text.insert("end", "\n\n")
        
        # LIDs
        self.output_text.insert("end", "🔤 Language IDs:\n", 'title')
        self.output_text.insert("end", f"   {result['lids']}\n\n")
        
        # ML
        self.output_text.insert("end", "🌐 English Translation (ML):\n", 'title')
        self.output_text.insert("end", f"   {result['ml']}\n\n")
        
        # MLI
        self.output_text.insert("end", "🗣️  Matrix Language Index (MLI):\n", 'title')
        self.output_text.insert("end", f"   {result['mli']}\n\n")
        
        # Sentiment
        self.output_text.insert("end", "😊 Sentiment:\n", 'title')
        self.output_text.insert("end", f"   {result['sentiment']}\n\n")
        
        # NER
        self.output_text.insert("end", "🏷️  Named Entities (NER):\n", 'title')
        if result['entities']:
            for word, entity_type in result['entities']:
                self.output_text.insert("end", f"   • {word} ", 'entity')
                self.output_text.insert("end", f"[{entity_type}]\n")
        else:
            self.output_text.insert("end", "   None\n")
        
        self.output_text.insert("end", "\n")
        
        # POS
        self.output_text.insert("end", "📌 Part-of-Speech Tags:\n", 'title')
        for i, (word, pos) in enumerate(result['pos_tags'][:10]):
            tag = pos if pos in self.pos_colors else 'default'
            self.output_text.insert("end", f"   {word} → ", tag)
            self.output_text.insert("end", f"{pos}\n")
        
        # Legend
        self.output_text.insert("end", "\n" + "─" * 60 + "\n")
        self.output_text.insert("end", "🎨 Color Legend:\n", 'title')
        legend_items = [
            ('NOUN', 'Noun'), ('VERB', 'Verb'), ('ADJ', 'Adjective'),
            ('ADV', 'Adverb'), ('PROPN', 'Proper Noun'), ('entity', 'Named Entity')
        ]
        for tag, label in legend_items:
            if tag in self.pos_colors or tag == 'entity':
                self.output_text.insert("end", f"   ● {label}  ", tag)
        
        self.status_label.config(text="✅ Analysis complete!", fg='#34d399')


if __name__ == "__main__":
    root = tk.Tk()
    app = HinglishAnalyzerGUI(root)
    root.mainloop()

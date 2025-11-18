"""Simple Hinglish Analyzer GUI - Opens immediately, loads models on demand"""
import tkinter as tk
from tkinter import scrolledtext, messagebox
import fasttext
import warnings

warnings.filterwarnings('ignore')

class SimpleHinglishGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌐 Hinglish Analyzer - Simple Version")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e1e')
        
        # Load only FastText model (fast)
        print("Loading FastText model...")
        self.lid_model = fasttext.load_model('lid.176.bin')
        print("✓ FastText loaded!")
        
        # Dictionaries
        self.hindi_dict = {
            'main': 'मैं', 'mein': 'मैं', 'theek': 'ठीक', 'thik': 'ठीक',
            'hoon': 'हूं', 'hun': 'हूं', 'kya': 'क्या', 'hai': 'है',
            'aaj': 'आज', 'kal': 'कल', 'yaar': 'यार', 'app': 'आप',
            'tum': 'तुम', 'hum': 'हम', 'kaise': 'कैसे', 'accha': 'अच्छा',
            'mera': 'मेरा', 'ka': 'का', 'ko': 'को', 'se': 'से',
            'me': 'में', 'naam': 'नाम', 'milte': 'मिलते', 'plan': 'प्लान'
        }
        
        self.hindi_to_english = {
            'मैं': 'I', 'ठीक': 'fine', 'हूं': 'am', 'क्या': 'what',
            'है': 'is', 'आज': 'today', 'कल': 'tomorrow', 'यार': 'friend',
            'आप': 'you', 'तुम': 'you', 'कैसे': 'how', 'अच्छा': 'good',
            'मेरा': 'my', 'का': 'of', 'को': 'to', 'से': 'from',
            'में': 'in', 'नाम': 'name', 'मिलते': 'meet', 'प्लान': 'plan'
        }
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="🌐 Hinglish Text Analyzer", 
                        font=('Arial', 20, 'bold'), bg='#1e1e1e', fg='#4a9eff')
        title.pack(pady=15)
        
        subtitle = tk.Label(self.root, text="Convert Romanized Hindi to Devanagari", 
                           font=('Arial', 10), bg='#1e1e1e', fg='#888888')
        subtitle.pack()
        
        # Input frame
        input_frame = tk.Frame(self.root, bg='#1e1e1e')
        input_frame.pack(pady=10, padx=20, fill='x')
        
        input_label = tk.Label(input_frame, text="📝 Enter Hinglish Text:", 
                              font=('Arial', 11, 'bold'), bg='#1e1e1e', fg='#e0e0e0')
        input_label.pack(anchor='w')
        
        self.input_text = tk.Text(input_frame, height=4, font=('Arial', 12),
                                 bg='#2d2d2d', fg='#e0e0e0',
                                 insertbackground='#e0e0e0', relief='flat',
                                 padx=10, pady=10)
        self.input_text.pack(fill='x', pady=5)
        self.input_text.insert("1.0", "main theek hoon how are you")
        
        # Analyze button
        analyze_btn = tk.Button(self.root, text="🔍 Analyze Text", command=self.analyze,
                               font=('Arial', 12, 'bold'), bg='#4a9eff', fg='white',
                               relief='flat', padx=30, pady=10, cursor='hand2')
        analyze_btn.pack(pady=10)
        
        # Output frame
        output_frame = tk.Frame(self.root, bg='#1e1e1e')
        output_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        output_label = tk.Label(output_frame, text="📄 Results:", 
                               font=('Arial', 11, 'bold'), bg='#1e1e1e', fg='#e0e0e0')
        output_label.pack(anchor='w')
        
        self.output_text = scrolledtext.ScrolledText(output_frame, font=('Arial', 11),
                                                     bg='#2d2d2d', fg='#e0e0e0',
                                                     relief='flat', padx=15, pady=15, wrap='word')
        self.output_text.pack(fill='both', expand=True, pady=5)
        
        # Status
        self.status_label = tk.Label(self.root, text="✅ Ready! Enter text and click Analyze", 
                                     font=('Arial', 10), bg='#1e1e1e', fg='#34d399')
        self.status_label.pack(pady=5)
        
        # Examples
        examples_frame = tk.Frame(self.root, bg='#1e1e1e')
        examples_frame.pack(pady=5)
        
        tk.Label(examples_frame, text="Try: ", font=('Arial', 9), 
                bg='#1e1e1e', fg='#888888').pack(side='left')
        
        examples = [
            "main theek hoon",
            "aaj ka plan kya hai",
            "how are you yaar"
        ]
        
        for ex in examples:
            btn = tk.Button(examples_frame, text=ex, command=lambda e=ex: self.load_example(e),
                           font=('Arial', 8), bg='#3d3d3d', fg='#e0e0e0',
                           relief='flat', padx=8, pady=4, cursor='hand2')
            btn.pack(side='left', padx=3)
    
    def load_example(self, text):
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", text)
        self.analyze()
    
    def detect_language(self, word):
        clean_word = word.lower().strip('.,!?;:')
        try:
            labels, scores = self.lid_model.predict(clean_word)
            lang = labels[0].replace("__label__", "")
            return 'en' if lang == 'en' and scores[0] > 0.6 else 'hi'
        except:
            return 'hi'
    
    def analyze(self):
        input_text = self.input_text.get("1.0", "end-1c").strip()
        if not input_text:
            messagebox.showwarning("Empty Input", "Please enter some text!")
            return
        
        self.status_label.config(text="🔄 Analyzing...", fg='#fbbf24')
        self.root.update()
        
        try:
            words = input_text.split()
            result_words = []
            lids = []
            english_translation = []
            
            for word in words:
                clean_word = word.strip('.,!?;:')
                punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ''
                lang = self.detect_language(clean_word)
                lids.append(lang)
                
                if lang == 'en':
                    result_words.append(word)
                    english_translation.append(word)
                else:
                    hindi_word = self.hindi_dict.get(clean_word.lower(), clean_word)
                    result_words.append(hindi_word + punctuation)
                    english_word = self.hindi_to_english.get(hindi_word, clean_word)
                    english_translation.append(english_word + punctuation)
            
            output = ' '.join(result_words)
            ml = ' '.join(english_translation)
            mli = 'Hindi' if lids.count('hi') >= lids.count('en') else 'English'
            
            # Display results
            self.output_text.delete("1.0", "end")
            
            self.output_text.insert("end", "="*70 + "\n")
            self.output_text.insert("end", "ANALYSIS RESULTS\n")
            self.output_text.insert("end", "="*70 + "\n\n")
            
            self.output_text.insert("end", "📝 Original Input:\n")
            self.output_text.insert("end", f"   {input_text}\n\n")
            
            self.output_text.insert("end", "📄 Converted Output (Devanagari):\n")
            self.output_text.insert("end", f"   {output}\n\n")
            
            self.output_text.insert("end", "🔤 Language IDs (per word):\n")
            self.output_text.insert("end", f"   {', '.join(lids)}\n\n")
            
            self.output_text.insert("end", "🌐 English Translation:\n")
            self.output_text.insert("end", f"   {ml}\n\n")
            
            self.output_text.insert("end", "🗣️ Matrix Language Index (Dominant Language):\n")
            self.output_text.insert("end", f"   {mli}\n\n")
            
            # Statistics
            hindi_count = lids.count('hi')
            english_count = lids.count('en')
            total = len(lids)
            
            self.output_text.insert("end", "="*70 + "\n")
            self.output_text.insert("end", "STATISTICS\n")
            self.output_text.insert("end", "="*70 + "\n\n")
            self.output_text.insert("end", f"   Total Words: {total}\n")
            self.output_text.insert("end", f"   Hindi Words: {hindi_count} ({hindi_count/total*100:.1f}%)\n")
            self.output_text.insert("end", f"   English Words: {english_count} ({english_count/total*100:.1f}%)\n\n")
            
            self.status_label.config(text="✅ Analysis complete!", fg='#34d399')
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_label.config(text=f"❌ Error: {str(e)}", fg='#ff6b6b')


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleHinglishGUI(root)
    print("\n✅ GUI is ready! Window should be visible now.")
    root.mainloop()

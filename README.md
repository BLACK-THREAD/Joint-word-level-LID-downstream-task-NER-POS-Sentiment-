# Joint-word-level-LID-downstream-task-NER-POS-Sentiment-

# 🌐 Hinglish Text Analyzer

A comprehensive NLP system for analyzing code-mixed Hinglish (Hindi-English) text with language identification, transliteration, sentiment analysis, named entity recognition, and part-of-speech tagging.

## 📦 Repository

**GitHub**: [https://github.com/BLACK-THREAD/Joint-word-level-LID-downstream-task-NER-POS-Sentiment-](https://github.com/BLACK-THREAD/Joint-word-level-LID-downstream-task-NER-POS-Sentiment-)

This repository contains a complete implementation of joint word-level Language Identification (LID) with downstream tasks including Named Entity Recognition (NER), Part-of-Speech (POS) tagging, and Sentiment Analysis for Hinglish text.

## 📋 Features

- **Language Identification (LID)**: Detects language for each word (Hindi/English)
- **Script Conversion**: Converts Romanized Hindi to Devanagari script
- **English Translation**: Translates Hindi words to English (Matrix Language)
- **Matrix Language Index (MLI)**: Identifies the dominant language in text
- **Sentiment Analysis**: Analyzes emotional tone of the text
- **Named Entity Recognition (NER)**: Identifies people, places, organizations
- **Part-of-Speech (POS) Tagging**: Tags grammatical categories
- **Interactive GUI**: Dark-themed Tkinter interface with color-coded results
- **Model Evaluation**: Performance metrics on COMI-LINGUA dataset

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 2GB+ free disk space (for models)
- Internet connection (for first-time model download)

### Step 1: Clone or Download the Project
```bash
cd /path/to/NLP101
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note for Windows users**: If you encounter issues with torch, install it separately:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Download FastText LID Model
The FastText model should already be in the project folder as `lid.176.bin`. If not, download it:
```bash
curl -o lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

## 📂 Project Structure

```
NLP101/
├── lid.176.bin                      # FastText language identification model
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── hinglish_analyzer_gui.py         # 🎨 Main GUI Application
├── complete_hinglish_pipeline.py    # Complete analysis pipeline
├── correct_hinglish_converter.py    # Hinglish to Devanagari converter
├── evaluate_model_final.py          # Model evaluation script
│
├── COMI-LINGUA/                     # Dataset folder
│   ├── LID_test.csv
│   ├── LID_train.csv
│   ├── MLI_test.csv
│   ├── MLI_train.csv
│   ├── NER_test.csv
│   ├── NER_train.csv
│   ├── POS_test.csv
│   └── POS_train.csv
│
└── report/                          # Evaluation results
    ├── evaluation_report.txt
    └── visualizations/
        ├── accuracy.jpg
        ├── distribution.jpg
        └── performance.jpg
```

## 🎯 Usage

### Option 1: GUI Application (Recommended)
Run the interactive graphical interface:
```bash
python hinglish_analyzer_gui.py
```

**How to use:**
1. Enter Hinglish text in the input box (e.g., "main theek hoon how are you")
2. Click "🔍 Analyze" button
3. View results with color-coded POS tags and entities

### Option 2: Command Line Pipeline
Run the complete analysis pipeline:
```bash
python complete_hinglish_pipeline.py
```

### Option 3: Evaluate Model
Test the model on COMI-LINGUA dataset:
```bash
python evaluate_model_final.py
```

## 📖 File Descriptions

### Core Files

**hinglish_analyzer_gui.py**
- Main GUI application with dark theme
- Interactive text analysis interface
- Color-coded output for POS tags and entities
- Real-time analysis with threading

**complete_hinglish_pipeline.py**
- Complete NLP pipeline combining all features
- Processes Hinglish text through all analysis stages
- Outputs: LID, ML, MLI, Sentiment, NER, POS

**correct_hinglish_converter.py**
- Converts Romanized Hindi to Devanagari script
- Keeps English words unchanged
- Uses dictionary-based + FastText LID approach
- Provides English translation (ML)

**evaluate_model_final.py**
- Evaluates model performance on COMI-LINGUA test set
- Generates accuracy metrics and visualizations
- Creates performance reports in `report/` folder

### Utility Files

**test_lid.py**
- Basic FastText LID testing
- Word-level language detection examples

**hinglish_converter.py**
- Simple transliteration using indic-transliteration library
- Basic Hinglish conversion functions

**hinglish_nlp_pipeline.py**
- NLP analysis functions (Sentiment, NER, POS)
- Model loading and inference

### Dataset

**COMI-LINGUA/**
- Code-Mixed Linguistic Annotation dataset
- Contains train/test files for:
  - LID (Language Identification)
  - MLI (Matrix Language Index)
  - NER (Named Entity Recognition)
  - POS (Part-of-Speech)
  - MT (Machine Translation)

## 🎨 GUI Features

### Color Legend
- **🟣 Purple**: Adjectives (ADJ)
- **🔴 Red**: Nouns (NOUN)
- **🔵 Blue**: Verbs (VERB)
- **🟡 Yellow**: Adverbs (ADV)
- **🟢 Green**: Named Entities (NER)
- **🌸 Pink**: Pronouns (PRON)
- **🟠 Orange**: Prepositions (ADP)

### Output Sections
1. **Input**: Original text
2. **Output**: Converted text with color-coded POS tags
3. **LIDs**: Language ID for each word
4. **ML**: English translation
5. **MLI**: Dominant language
6. **Sentiment**: Emotional analysis
7. **NER**: Named entities
8. **POS**: Part-of-speech tags

## 📊 Model Performance

Evaluated on COMI-LINGUA test dataset:
- **Accuracy**: 68.79%
- **Test Samples**: 100 sentences (2,009 words)
- **Languages**: Hindi (68.6%), English (31.4%)

## 🔧 Troubleshooting

### Issue: Models taking too long to load
**Solution**: First run downloads models from HuggingFace. Subsequent runs will be faster.

### Issue: Out of memory error
**Solution**: Close other applications or reduce batch size in code.

### Issue: GUI not opening
**Solution**: Ensure tkinter is installed:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (usually pre-installed)
brew install python-tk

# Windows (usually pre-installed with Python)
```

### Issue: CUDA/GPU errors
**Solution**: The code uses CPU by default. If you have GPU issues, ensure PyTorch CPU version is installed.

## 💡 Example Usage

### Input:
```
main theek hoon how are you
```

### Output:
```
📝 Input: main theek hoon how are you
📄 Output: मैं ठीक हूं how are you
🔤 LIDs: ['hi', 'hi', 'hi', 'en', 'en', 'en']
🌐 ML: I fine am how are you
🗣️ MLI: Hindi
😊 Sentiment: neutral (0.4466)
🏷️ NER: None
📌 POS: main→ADJ, thee→NOUN, ##k→NOUN, ho→INTJ, ##on→NOUN, how→ADV, are→AUX, you→PRON
```

## 📝 Requirements

- Python 3.8+
- 2GB RAM minimum (4GB recommended)
- 3GB disk space (for models and dataset)
- Internet connection (first run only)

## 🤝 Credits

- **FastText**: Facebook AI Research
- **Transformers**: HuggingFace
- **COMI-LINGUA Dataset**: LINGO Lab, IIT Gandhinagar
- **Models Used**:
  - cardiffnlp/twitter-xlm-roberta-base-sentiment
  - Davlan/xlm-roberta-base-ner-hrl
  - vblagoje/bert-english-uncased-finetuned-pos

## 📄 License

This project uses various open-source models and datasets. Please refer to their respective licenses.

## 🐛 Known Issues

- POS tagging may not be accurate for Romanized Hindi words
- NER works better with proper nouns in English
- Sentiment analysis is optimized for social media text

## 🚀 Future Improvements

- Add more Hindi words to dictionary
- Improve Romanized Hindi detection
- Add support for more Indian languages
- Implement custom fine-tuned models for Hinglish

---

**Made with ❤️ for Hinglish NLP**
#   J o i n t - w o r d - l e v e l - L I D - d o w n s t r e a m - t a s k - N E R - P O S - S e n t i m e n t - 
 
 #   J o i n t - w o r d - l e v e l - L I D - d o w n s t r e a m - t a s k - N E R - P O S - S e n t i m e n t - 
 
 #   J o i n t - w o r d - l e v e l - L I D - d o w n s t r e a m - t a s k - N E R - P O S - S e n t i m e n t - 
 
 
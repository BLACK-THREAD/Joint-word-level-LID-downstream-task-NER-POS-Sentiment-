import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import fasttext
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 70)
print("COMI-LINGUA MODEL EVALUATION")
print("=" * 70)
print()

# Create report directories
os.makedirs('report/tests', exist_ok=True)
os.makedirs('report/visualizations', exist_ok=True)

# Load FastText LID model
print("Loading FastText LID model...")
lid_model = fasttext.load_model('lid.176.bin')
print("✓ Model loaded\n")

# Hindi word dictionary for LID
hindi_dict = {
    'main': 'hi', 'mein': 'hi', 'app': 'hi', 'aap': 'hi',
    'tum': 'hi', 'hum': 'hi', 'woh': 'hi', 'wo': 'hi',
    'hai': 'hi', 'hain': 'hi', 'ho': 'hi', 'hoon': 'hi',
    'kya': 'hi', 'kyu': 'hi', 'kab': 'hi', 'kahan': 'hi',
    'kaun': 'hi', 'kaise': 'hi', 'kaisa': 'hi',
    'theek': 'hi', 'accha': 'hi', 'haal': 'hi',
    'kal': 'hi', 'aaj': 'hi', 'mera': 'hi',
    'ka': 'hi', 'ko': 'hi', 'se': 'hi', 'me': 'hi',
}

english_words = {
    'bro', 'dude', 'hello', 'hi', 'hey', 'bye', 'okay', 'ok',
    'how', 'are', 'you', 'i', 'am', 'is', 'was', 'were',
    'good', 'bad', 'nice', 'great', 'awesome', 'cool',
    'the', 'a', 'an', 'and', 'or', 'but',
}


def predict_lid(word):
    """Predict language ID for a word"""
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


def predict_mli(text):
    """Predict Matrix Language Index (main language)"""
    words = text.split()
    lids = [predict_lid(word) for word in words]
    hindi_count = lids.count('hi')
    english_count = lids.count('en')
    return 'Hindi' if hindi_count >= english_count else 'English'


print("=" * 70)
print("1. EVALUATING LID (Language Identification)")
print("=" * 70)

# Load LID test data
print("Loading LID test data...")
lid_test = pd.read_csv('COMI-LINGUA/LID_test.csv')
print(f"Loaded {len(lid_test)} test samples\n")

# Sample for faster evaluation (use first 1000 samples)
lid_sample = lid_test.head(1000)

print("Running predictions...")
y_true_lid = []
y_pred_lid = []

for idx, row in tqdm(lid_sample.iterrows(), total=len(lid_sample), desc="LID Evaluation"):
    if 'word' in row and 'label' in row:
        word = str(row['word'])
        true_label = str(row['label']).lower()
        pred_label = predict_lid(word)
        
        # Normalize labels
        if true_label in ['hindi', 'hi', 'hin']:
            true_label = 'hi'
        elif true_label in ['english', 'en', 'eng']:
            true_label = 'en'
        
        y_true_lid.append(true_label)
        y_pred_lid.append(pred_label)

# Calculate metrics
lid_accuracy = accuracy_score(y_true_lid, y_pred_lid)
lid_precision, lid_recall, lid_f1, _ = precision_recall_fscore_support(y_true_lid, y_pred_lid, average='weighted', zero_division=0)

print(f"\n✓ LID Evaluation Complete")
print(f"  Accuracy:  {lid_accuracy:.4f}")
print(f"  Precision: {lid_precision:.4f}")
print(f"  Recall:    {lid_recall:.4f}")
print(f"  F1-Score:  {lid_f1:.4f}\n")

# Confusion Matrix for LID
cm_lid = confusion_matrix(y_true_lid, y_pred_lid)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lid, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['English', 'Hindi'], 
            yticklabels=['English', 'Hindi'])
plt.title('LID Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('report/visualizations/lid_confusion_matrix.jpg', dpi=300, bbox_inches='tight')
print("✓ Saved: report/visualizations/lid_confusion_matrix.jpg")
plt.close()

# LID Accuracy Bar Chart
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [lid_accuracy, lid_precision, lid_recall, lid_f1]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
plt.ylim(0, 1.0)
plt.title('LID Model Performance Metrics', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('report/visualizations/lid_metrics.jpg', dpi=300, bbox_inches='tight')
print("✓ Saved: report/visualizations/lid_metrics.jpg")
plt.close()


print("\n" + "=" * 70)
print("2. EVALUATING MLI (Matrix Language Index)")
print("=" * 70)

# Load MLI test data
print("Loading MLI test data...")
mli_test = pd.read_csv('COMI-LINGUA/MLI_test.csv')
print(f"Loaded {len(mli_test)} test samples\n")

# Sample for faster evaluation
mli_sample = mli_test.head(500)

print("Running predictions...")
y_true_mli = []
y_pred_mli = []

for idx, row in tqdm(mli_sample.iterrows(), total=len(mli_sample), desc="MLI Evaluation"):
    if 'text' in row and 'label' in row:
        text = str(row['text'])
        true_label = str(row['label'])
        pred_label = predict_mli(text)
        
        # Normalize labels
        if 'hindi' in true_label.lower() or 'hi' in true_label.lower():
            true_label = 'Hindi'
        elif 'english' in true_label.lower() or 'en' in true_label.lower():
            true_label = 'English'
        
        y_true_mli.append(true_label)
        y_pred_mli.append(pred_label)

# Calculate metrics
mli_accuracy = accuracy_score(y_true_mli, y_pred_mli)
mli_precision, mli_recall, mli_f1, _ = precision_recall_fscore_support(y_true_mli, y_pred_mli, average='weighted', zero_division=0)

print(f"\n✓ MLI Evaluation Complete")
print(f"  Accuracy:  {mli_accuracy:.4f}")
print(f"  Precision: {mli_precision:.4f}")
print(f"  Recall:    {mli_recall:.4f}")
print(f"  F1-Score:  {mli_f1:.4f}\n")

# Confusion Matrix for MLI
cm_mli = confusion_matrix(y_true_mli, y_pred_mli)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mli, annot=True, fmt='d', cmap='Greens',
            xticklabels=['English', 'Hindi'],
            yticklabels=['English', 'Hindi'])
plt.title('MLI Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('report/visualizations/mli_confusion_matrix.jpg', dpi=300, bbox_inches='tight')
print("✓ Saved: report/visualizations/mli_confusion_matrix.jpg")
plt.close()

# MLI Accuracy Bar Chart
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [mli_accuracy, mli_precision, mli_recall, mli_f1]
colors = ['#27ae60', '#16a085', '#d35400', '#c0392b']
bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
plt.ylim(0, 1.0)
plt.title('MLI Model Performance Metrics', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('report/visualizations/mli_metrics.jpg', dpi=300, bbox_inches='tight')
print("✓ Saved: report/visualizations/mli_metrics.jpg")
plt.close()


print("\n" + "=" * 70)
print("3. GENERATING COMPARISON REPORT")
print("=" * 70)

# Overall comparison
plt.figure(figsize=(14, 8))
x = np.arange(4)
width = 0.35

lid_scores = [lid_accuracy, lid_precision, lid_recall, lid_f1]
mli_scores = [mli_accuracy, mli_precision, mli_recall, mli_f1]

bars1 = plt.bar(x - width/2, lid_scores, width, label='LID', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = plt.bar(x + width/2, mli_scores, width, label='MLI', color='#2ecc71', alpha=0.8, edgecolor='black')

plt.xlabel('Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.title('Model Performance Comparison: LID vs MLI', fontsize=16, fontweight='bold')
plt.xticks(x, ['Accuracy', 'Precision', 'Recall', 'F1-Score'], fontsize=12)
plt.ylim(0, 1.0)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('report/visualizations/overall_comparison.jpg', dpi=300, bbox_inches='tight')
print("✓ Saved: report/visualizations/overall_comparison.jpg")
plt.close()

# Save detailed report
report_text = f"""
{'=' * 70}
COMI-LINGUA MODEL EVALUATION REPORT
{'=' * 70}

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 70}
1. LID (Language Identification) Results
{'=' * 70}

Test Samples: {len(lid_sample)}

Metrics:
  - Accuracy:  {lid_accuracy:.4f} ({lid_accuracy*100:.2f}%)
  - Precision: {lid_precision:.4f}
  - Recall:    {lid_recall:.4f}
  - F1-Score:  {lid_f1:.4f}

{'=' * 70}
2. MLI (Matrix Language Index) Results
{'=' * 70}

Test Samples: {len(mli_sample)}

Metrics:
  - Accuracy:  {mli_accuracy:.4f} ({mli_accuracy*100:.2f}%)
  - Precision: {mli_precision:.4f}
  - Recall:    {mli_recall:.4f}
  - F1-Score:  {mli_f1:.4f}

{'=' * 70}
3. Summary
{'=' * 70}

The model shows:
- LID Accuracy: {lid_accuracy*100:.2f}%
- MLI Accuracy: {mli_accuracy*100:.2f}%

Overall Performance: {'Good' if (lid_accuracy + mli_accuracy) / 2 > 0.7 else 'Needs Improvement'}

Generated Visualizations:
  ✓ report/visualizations/lid_confusion_matrix.jpg
  ✓ report/visualizations/lid_metrics.jpg
  ✓ report/visualizations/mli_confusion_matrix.jpg
  ✓ report/visualizations/mli_metrics.jpg
  ✓ report/visualizations/overall_comparison.jpg

{'=' * 70}
"""

with open('report/evaluation_report.txt', 'w') as f:
    f.write(report_text)

print("\n✓ Saved: report/evaluation_report.txt")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print(f"\nResults saved in:")
print(f"  📁 report/visualizations/ (5 JPG images)")
print(f"  📄 report/evaluation_report.txt")
print("\n" + "=" * 70)

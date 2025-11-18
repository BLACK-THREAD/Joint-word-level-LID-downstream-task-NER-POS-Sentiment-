import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import fasttext
import warnings
import ast
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

print("=" * 70)
print("HINGLISH MODEL EVALUATION - COMI-LINGUA DATASET")
print("=" * 70)
print()

# Create directories
os.makedirs('report/tests', exist_ok=True)
os.makedirs('report/visualizations', exist_ok=True)

# Load FastText model
print("Loading FastText LID model...")
lid_model = fasttext.load_model('lid.176.bin')
print("✓ Model loaded\n")

def predict_lid(word):
    """Predict language ID"""
    try:
        labels, scores = lid_model.predict(word.lower())
        lang = labels[0].replace("__label__", "")
        return 'hi' if lang == 'hi' else 'en'
    except:
        return 'hi'

# Load LID test data
print("Loading LID test data...")
lid_df = pd.read_csv('COMI-LINGUA/LID_test.csv')
print(f"Total samples: {len(lid_df)}\n")

# Evaluate on sample
sample_size = 100
y_true = []
y_pred = []

print(f"Evaluating {sample_size} samples...")
for idx in range(min(sample_size, len(lid_df))):
    try:
        row = lid_df.iloc[idx]
        # Parse annotations (convert string to list of dicts)
        annotations_str = row['Annotated by: Annotator 1']
        annotations = ast.literal_eval(annotations_str)
        
        for item in annotations:
            word = item['key']
            true_label = item['value']
            
            # Skip 'ot' (other) labels and punctuation
            if true_label == 'ot' or len(word.strip()) == 0:
                continue
            
            pred_label = predict_lid(word)
            y_true.append(true_label)
            y_pred.append(pred_label)
    except Exception as e:
        continue

print(f"✓ Evaluated {len(y_true)} words\n")

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
correct = sum([1 for t, p in zip(y_true, y_pred) if t == p])

print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Total words:  {len(y_true)}")
print(f"Correct:      {correct}")
print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print("=" * 70)
print()

# Count distribution
from collections import Counter
true_dist = Counter(y_true)
pred_dist = Counter(y_pred)

print("Language Distribution:")
print(f"  Ground Truth - EN: {true_dist.get('en', 0)}, HI: {true_dist.get('hi', 0)}")
print(f"  Predicted    - EN: {pred_dist.get('en', 0)}, HI: {pred_dist.get('hi', 0)}")
print()

# Generate visualizations
print("Generating visualizations...")

# 1. Accuracy Bar
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(['Accuracy'], [accuracy], color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2, width=0.4)
ax.set_ylim(0, 1.0)
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('LID Model Accuracy', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.text(0, accuracy + 0.05, f'{accuracy:.3f}\n({accuracy*100:.1f}%)',
        ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('report/visualizations/accuracy.jpg', dpi=150, bbox_inches='tight')
print("✓ Saved: accuracy.jpg")
plt.close()

# 2. Distribution comparison
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2)
width = 0.35
true_counts = [true_dist.get('en', 0), true_dist.get('hi', 0)]
pred_counts = [pred_dist.get('en', 0), pred_dist.get('hi', 0)]

ax.bar(x - width/2, true_counts, width, label='Ground Truth', color='#3498db', alpha=0.8, edgecolor='black')
ax.bar(x + width/2, pred_counts, width, label='Predicted', color='#2ecc71', alpha=0.8, edgecolor='black')

ax.set_xlabel('Language', fontsize=12, fontweight='bold')
ax.set_ylabel('Word Count', fontsize=12, fontweight='bold')
ax.set_title('Language Distribution', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['English', 'Hindi'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/visualizations/distribution.jpg', dpi=150, bbox_inches='tight')
print("✓ Saved: distribution.jpg")
plt.close()

# 3. Performance gauge
fig, ax = plt.subplots(figsize=(6, 4))
color = '#2ecc71' if accuracy > 0.75 else '#f39c12' if accuracy > 0.5 else '#e74c3c'
ax.barh(['LID Model'], [accuracy], color=color, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_xlim(0, 1.0)
ax.set_xlabel('Accuracy Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.text(accuracy + 0.05, 0, f'{accuracy*100:.1f}%', va='center', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/visualizations/performance.jpg', dpi=150, bbox_inches='tight')
print("✓ Saved: performance.jpg")
plt.close()

# Save report
report = f"""
{'=' * 70}
HINGLISH LID MODEL - EVALUATION REPORT
{'=' * 70}

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: COMI-LINGUA Test Set

{'=' * 70}
RESULTS
{'=' * 70}

Samples Evaluated: {sample_size} sentences
Total Words: {len(y_true)}
Correct Predictions: {correct}

Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)

Language Distribution (Ground Truth):
  - English: {true_dist.get('en', 0)} words ({true_dist.get('en', 0)/len(y_true)*100:.1f}%)
  - Hindi:   {true_dist.get('hi', 0)} words ({true_dist.get('hi', 0)/len(y_true)*100:.1f}%)

Language Distribution (Predicted):
  - English: {pred_dist.get('en', 0)} words ({pred_dist.get('en', 0)/len(y_true)*100:.1f}%)
  - Hindi:   {pred_dist.get('hi', 0)} words ({pred_dist.get('hi', 0)/len(y_true)*100:.1f}%)

{'=' * 70}
VISUALIZATIONS
{'=' * 70}

✓ report/visualizations/accuracy.jpg
✓ report/visualizations/distribution.jpg
✓ report/visualizations/performance.jpg

{'=' * 70}
"""

with open('report/evaluation_report.txt', 'w') as f:
    f.write(report)

print("✓ Saved: evaluation_report.txt")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print(f"\n📊 Accuracy: {accuracy*100:.2f}%")
print(f"📁 Output: report/visualizations/ & report/evaluation_report.txt")
print("=" * 70)

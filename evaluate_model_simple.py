import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import fasttext
import warnings
import json
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

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

# Simple LID prediction
def predict_lid_simple(word):
    """Simple LID prediction"""
    try:
        labels, scores = lid_model.predict(word.lower())
        lang = labels[0].replace("__label__", "")
        return 'hi' if lang == 'hi' else 'en'
    except:
        return 'hi'

# Load and parse LID test data
print("Loading LID test data...")
lid_df = pd.read_csv('COMI-LINGUA/LID_test.csv')
print(f"Total samples: {len(lid_df)}")

# Sample 200 sentences for evaluation
sample_size = 200
lid_sample = lid_df.head(sample_size)

print(f"Evaluating {sample_size} samples...\n")

# Evaluate LID
y_true = []
y_pred = []
correct = 0
total = 0

for idx, row in lid_sample.iterrows():
    try:
        # Parse annotated tags (using first annotator)
        annotations = json.loads(row['Annotated by: Annotator 1'])
        
        for item in annotations:
            word = item['key']
            true_label = item['value']
            
            # Skip 'ot' (other) labels
            if true_label == 'ot':
                continue
            
            pred_label = predict_lid_simple(word)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            total += 1
            
            if pred_label == true_label:
                correct += 1
    except:
        continue

# Calculate metrics
accuracy = correct / total if total > 0 else 0
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted', zero_division=0
)

print("=" * 70)
print("LID EVALUATION RESULTS")
print("=" * 70)
print(f"Total words evaluated: {total}")
print(f"Correct predictions:   {correct}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=" * 70)
print()

# Create visualizations
print("Generating visualizations...")

# 1. Accuracy Bar Chart
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylim(0, 1.0)
plt.title('Hinglish LID Model Performance', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{value:.3f}\n({value*100:.1f}%)',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('report/visualizations/lid_performance.jpg', dpi=300, bbox_inches='tight')
print("✓ Saved: report/visualizations/lid_performance.jpg")
plt.close()

# 2. Language Distribution
from collections import Counter
true_dist = Counter(y_true)
pred_dist = Counter(y_pred)

plt.figure(figsize=(12, 6))
x = np.arange(2)
width = 0.35

true_counts = [true_dist.get('en', 0), true_dist.get('hi', 0)]
pred_counts = [pred_dist.get('en', 0), pred_dist.get('hi', 0)]

bars1 = plt.bar(x - width/2, true_counts, width, label='Ground Truth', 
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = plt.bar(x + width/2, pred_counts, width, label='Predicted',
                color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

plt.xlabel('Language', fontsize=14, fontweight='bold')
plt.ylabel('Word Count', fontsize=14, fontweight='bold')
plt.title('Language Distribution: Ground Truth vs Predicted', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, ['English', 'Hindi'], fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 20,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('report/visualizations/language_distribution.jpg', dpi=300, bbox_inches='tight')
print("✓ Saved: report/visualizations/language_distribution.jpg")
plt.close()

# 3. Accuracy Gauge
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})

theta = np.linspace(0, np.pi, 100)
r = np.ones(100)

# Background
ax.plot(theta, r, color='lightgray', linewidth=20, alpha=0.3)

# Accuracy arc
acc_theta = np.linspace(0, np.pi * accuracy, 100)
if accuracy < 0.5:
    color = '#e74c3c'
elif accuracy < 0.75:
    color = '#f39c12'
else:
    color = '#2ecc71'

ax.plot(acc_theta, r[:len(acc_theta)], color=color, linewidth=20)

ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['polar'].set_visible(False)

plt.text(0, 0, f'{accuracy*100:.1f}%', ha='center', va='center',
         fontsize=48, fontweight='bold', color=color)
plt.text(0, -0.3, 'Overall Accuracy', ha='center', va='center',
         fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('report/visualizations/accuracy_gauge.jpg', dpi=300, bbox_inches='tight')
print("✓ Saved: report/visualizations/accuracy_gauge.jpg")
plt.close()

# 4. Performance Summary
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

summary_text = f"""
HINGLISH LID MODEL - EVALUATION SUMMARY

Dataset: COMI-LINGUA (Test Set)
Samples Evaluated: {sample_size} sentences
Total Words: {total}

PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Accuracy:   {accuracy:.4f}  ({accuracy*100:.2f}%)
Precision:  {precision:.4f}
Recall:     {recall:.4f}
F1-Score:   {f1:.4f}

LANGUAGE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ground Truth:
  • English: {true_dist.get('en', 0)} words ({true_dist.get('en', 0)/total*100:.1f}%)
  • Hindi:   {true_dist.get('hi', 0)} words ({true_dist.get('hi', 0)/total*100:.1f}%)

Predicted:
  • English: {pred_dist.get('en', 0)} words ({pred_dist.get('en', 0)/total*100:.1f}%)
  • Hindi:   {pred_dist.get('hi', 0)} words ({pred_dist.get('hi', 0)/total*100:.1f}%)

MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rating: {'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.75 else 'Fair' if accuracy > 0.6 else 'Needs Improvement'}

The model successfully identifies language at word-level in code-mixed
Hinglish text with {accuracy*100:.1f}% accuracy.
"""

plt.text(0.5, 0.5, summary_text, ha='center', va='center',
         fontsize=12, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('report/visualizations/summary_report.jpg', dpi=300, bbox_inches='tight')
print("✓ Saved: report/visualizations/summary_report.jpg")
plt.close()

# Save text report
report_content = f"""
{'=' * 70}
HINGLISH LID MODEL - EVALUATION REPORT
{'=' * 70}

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: COMI-LINGUA Test Set

{'=' * 70}
EVALUATION RESULTS
{'=' * 70}

Samples Evaluated: {sample_size} sentences
Total Words: {total}
Correct Predictions: {correct}

Metrics:
  - Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
  - Precision: {precision:.4f}
  - Recall:    {recall:.4f}
  - F1-Score:  {f1:.4f}

Language Distribution (Ground Truth):
  - English: {true_dist.get('en', 0)} words ({true_dist.get('en', 0)/total*100:.1f}%)
  - Hindi:   {true_dist.get('hi', 0)} words ({true_dist.get('hi', 0)/total*100:.1f}%)

Language Distribution (Predicted):
  - English: {pred_dist.get('en', 0)} words ({pred_dist.get('en', 0)/total*100:.1f}%)
  - Hindi:   {pred_dist.get('hi', 0)} words ({pred_dist.get('hi', 0)/total*100:.1f}%)

{'=' * 70}
VISUALIZATIONS GENERATED
{'=' * 70}

✓ report/visualizations/lid_performance.jpg
✓ report/visualizations/language_distribution.jpg
✓ report/visualizations/accuracy_gauge.jpg
✓ report/visualizations/summary_report.jpg

{'=' * 70}
"""

with open('report/evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("✓ Saved: report/evaluation_report.txt")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print(f"\n📊 Results:")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  Total Words Evaluated: {total}")
print(f"\n📁 Output:")
print(f"  report/visualizations/ (4 JPG images)")
print(f"  report/evaluation_report.txt")
print("=" * 70)

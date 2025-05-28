import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
predictions_df = pd.read_csv('test_predictions.csv')
combined_df = pd.read_csv('testdf.csv')

# Rename columns and add actuals
predictions_df.rename(columns={predictions_df.columns[0]: 'predicted_classes'}, inplace=True)
predictions_df['actual_class'] = combined_df['ClassId']
predictions_df['file_name'] = combined_df['Path']
predictions_df['correct'] = predictions_df['predicted_classes'] == predictions_df['actual_class']

# Save result
predictions_df.to_csv('test_result.csv', index=False)

# Overall accuracy
overall_accuracy = accuracy_score(predictions_df['actual_class'], predictions_df['predicted_classes'])

# Classification report (precision, recall, f1-score, support)
report = classification_report(
    predictions_df['actual_class'],
    predictions_df['predicted_classes'],
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classwise_metrics.csv')

# Save stats to stats.txt
with open('statstest.txt', 'w') as f:
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(
        predictions_df['actual_class'],
        predictions_df['predicted_classes']
    ))

# Confusion matrix
cm = confusion_matrix(predictions_df['actual_class'], predictions_df['predicted_classes'])
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('TEST_confusion_matrix_annotated.png')
plt.close()

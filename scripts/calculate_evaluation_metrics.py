import os
import json
from src.config import DATASET_PATH
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

save_name = os.path.join(DATASET_PATH, 'predictions.json')
print(save_name)
with open(save_name, 'r') as f:
    data = json.load(f)


y_true = []
y_pred = []

for sent in data:
    for token in sent:
        y_true.append(token[1])
        y_pred.append(token[2])

report = classification_report(y_true, y_pred)

classification_report_name = os.path.join(DATASET_PATH, 'classification_report.txt')
# Save to a .txt file
with open(classification_report_name, 'w') as f:
    f.write(report)

class_labels = list({i for i in y_true})
cm = confusion_matrix(y_true, y_pred, labels = class_labels)
import pandas as pd

# Create a DataFrame for better formatting
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)

# Save to CSV
cm_report_name = os.path.join(DATASET_PATH, "confusion_matrix.csv")
df_cm.to_csv(cm_report_name)
##############################################################

# Get the unique labels (sorted to keep consistent order)
labels = sorted(list(set(y_true) | set(y_pred)))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Normalize row-wise
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

# Create DataFrame
df_cm = pd.DataFrame(cm_normalized, index=labels, columns=labels)

# Add support column (sum of true instances per class)
df_cm['support'] = cm.sum(axis=1)

# Optional: round for readability
df_cm = df_cm.round(2)

cm_report_name = os.path.join(DATASET_PATH, "confusion_matrix_normalized.xlsx")
df_cm = df_cm[list(df_cm.index)]
df_cm.to_excel(cm_report_name)

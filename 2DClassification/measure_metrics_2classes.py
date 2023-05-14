import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import collections
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support

# Load the CSV file
day = 'D14' # Can be D2, D6, D14
num_exp = 4
ocsv_file = '/home/youness/data/EffNet/2Classes_Classification/Results_1by1/' + 'efficientnet7_Class_' + day + '_exp' + str(num_exp) + '_' + str(10) + '_clean_images.csv'
df = pd.read_csv(ocsv_file)

# Get the ground truth labels and predicted labels
y_test = df.iloc[1:, 1]
predicted_labels = df.iloc[1:, 2]

# Calculate and print balanced accuracy score
balanced_accuracy = balanced_accuracy_score(y_test, predicted_labels)
print("Balanced Accuracy:", balanced_accuracy)

# Calculate and print the frequency of ground truth labels
frequency_gt = collections.Counter(y_test.values.tolist())
print("Ground Truth Frequency:", dict(frequency_gt))

# Calculate and print the frequency of predicted labels
frequency_pred = collections.Counter(predicted_labels.values.tolist())
print("Predicted Labels Frequency:", dict(frequency_pred))

# Define target names for classification report
target_names = ['day 0', 'day 14'] # ['day 0', 'day target']

# Print classification report
print("Classification Report:")
print(classification_report(y_test, predicted_labels, target_names=target_names))

# Calculate precision, recall, and F1-score
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, predicted_labels, average='weighted')
print('Precision={}, Recall={}, F1-score={}'.format(precision, recall, f1_score))

# Create and display the confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted_labels)
print("Confusion Matrix:")
print(cnf_matrix)
plt.imshow(cnf_matrix)
plt.show()

# Calculate True Positives (TP), False Negatives (FN), True Negatives (TN), and False Positives (FP)
TP = cnf_matrix[0][0]
FP = cnf_matrix[0][1]
FN = cnf_matrix[1][0]
TN = cnf_matrix[1][1]

# Calculate sensitivity, specificity, and accuracy
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)
print("True Positives (TP):", TP)
print("False Negatives (FN):", FN)
print("Accuracy:", accuracy)
print("Sensitivity (True Positive Rate):", sensitivity)
print("Specificity (True Negative Rate):", specificity)

# Experiment 7
# Confusion Matrix

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Actual values
actual = np.array([
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Dog',
    'Not Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Predicted values
predicted = np.array([
    'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog',
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Compute confusion matrix
cm = confusion_matrix(actual, predicted)

# Plot heatmap
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Dog', 'Not Dog'],
    yticklabels=['Dog', 'Not Dog']
)

# Label adjustments
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

# Classification report
print(classification_report(actual, predicted))




# Confusion Matrix (Multiclass)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# True labels
y_true = ['Cat'] * 10 + ['Dog'] * 12 + ['Horse'] * 10
print(y_true[:5])

# Predicted labels
y_pred = (
    ['Cat'] * 8 +
    ['Dog'] +
    ['Horse'] +
    ['Cat'] * 2 +
    ['Dog'] * 10 +
    ['Horse'] * 8 +
    ['Dog'] * 2
)
print(y_pred[:5])

# Class labels
classes = ['Cat', 'Dog', 'Horse']

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Greens)

# Title and axis adjustments
plt.title("Confusion Matrix")
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()

plt.show()

# Classification report
print(classification_report(y_true, y_pred))
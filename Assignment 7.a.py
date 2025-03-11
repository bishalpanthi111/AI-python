import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Step 0: Read data into a pandas dataframe
data = pd.read_csv('data_banknote_authentication.csv')

# Step 1: Pick the column named "class" as target variable y and all other columns as feature variables X
X = data.drop('class', axis=1)
y = data['class']

# Step 2: Split the data into training and testing sets with 80/20 ratio and random_state=20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Step 3: Use support vector classifier with linear kernel to fit to the training data
svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train, y_train)

# Step 4: Predict on the testing data and compute the confusion matrix and classification report
y_pred_linear = svc_linear.predict(X_test)
print("Confusion Matrix (Linear Kernel):")
print(confusion_matrix(y_test, y_pred_linear))
print("Classification Report (Linear Kernel):")
print(classification_report(y_test, y_pred_linear))

# Step 5: Repeat steps 3 and 4 for the radial basis function kernel
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)
y_pred_rbf = svc_rbf.predict(X_test)
print("Confusion Matrix (RBF Kernel):")
print(confusion_matrix(y_test, y_pred_rbf))
print("Classification Report (RBF Kernel):")
print(classification_report(y_test, y_pred_rbf))

# Step 6: Compare the two SVM models
# You can compare the accuracy, precision, recall, and F1-score from the classification reports
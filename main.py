# Import Necessary Packages

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

# Load the dataset from the csv file using pandas and take a quick look
data = pd.read_csv("creditcard.csv")
print('Data Head: ')
print(data.head())

# Initial Data Analysis
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
print('Overall Shape:')
print(data.shape)
print('Overall Description:')
print(data.describe())
outlierFraction = len(fraud)/float(len(valid))
print('Outlier Fraction: ', outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
print('Fraudulent Transactions Amount Details')
print(fraud.Amount.describe())
print('Valid Transactions Amount Details')
print(valid.Amount.describe())

# Correlation matrix
corrMat = data.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrMat, vmax=0.8, square=True)
plt.show()

# Segmenting the X and the Y from the dataset
X = data.drop(['Class'], axis=1)
Y = data["Class"]
print(X.shape)
print(Y.shape)
# Getting just the values for the sake of processing
xData = X.values
yData = Y.values


# Train-Test Split
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)
# Train Random Forest Binary Classifier
# rfc = RandomForestClassifier(class_weight={0: 1, 1: 581.4})
# rfc = RandomForestClassifier(class_weight='balanced')
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)

# Evaluating the classifier
print("The model used is Random Forest classifier")
acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))
prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))
rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))
f1 = f1_score(yTest, yPred)
print("The F1-Score is {}".format(f1))
MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is {}".format(MCC))

conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'], annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

#import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv("30 Days Challenge/har_data.csv")

# preprocess the dataset
X = df.drop('Activity' , axis=1)
y= df['Activity']

#split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Train the Random Forest Classifer

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy, precision , recall, f1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall =recall_score(y_test, y_pred, average='macro')
f1= f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1_score: {f1*100:.2f}%")

#Visualise the confusion matrix using Seaborns heatmap

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
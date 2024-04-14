import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df1 = pd.read_csv('llamaguard/results/accuracy_1000samp_data.csv')
df2 = pd.read_csv('nemo/predictions_nemo.csv')

# Renaming columns
df1.rename(columns={'expected': 'True', 'predicted': 'Predicted1'}, inplace=True)
df2.rename(columns={'Predicted': 'Predicted2'}, inplace=True)

# Combining datasets
df_combined = pd.concat([df1['Predicted1'], df2['Predicted2'], df1['True']], axis=1)

# Initialize the label encoder
encoder = LabelEncoder()

# Encode the predictions and true labels
df_combined['Predicted1'] = encoder.fit_transform(df_combined['Predicted1'])
df_combined['Predicted2'] = encoder.transform(df_combined['Predicted2'])
df_combined['True'] = encoder.transform(df_combined['True'])

X = df_combined[['Predicted1', 'Predicted2']]
y = df_combined['True']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))


conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", conf_matrix)

# Get the coefficients (weights) assigned to each feature
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients for llama:", coefficients[0][0])
print("Coefficients for nemo:", coefficients[0][1])
print("Intercept:", intercept[0])

print(df_combined['Predicted1'])
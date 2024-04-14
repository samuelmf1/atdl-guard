import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df1 = pd.read_csv('llamaguard/results/accuracy_1000samp_data.csv')
df2 = pd.read_csv('nemo/predictions_nemo.csv')

# Prepare the combined DataFrame
data = pd.DataFrame({
    'Prompt': df1['prompt'],
    'Llama': df1['predicted'],
    'Nemo': df2['Predicted'],
    'True': df1['expected']
})

# Split the data into training and testing sets (80/20 split)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to apply Model 1 logic
def model_1(row):
    if row['Llama'] == 'safe':
        return row['Nemo']
    else:
        return row['Llama']

# Function to apply Model 2 logic
def model_2(row):
    if row['Nemo'] == 'safe':
        return row['Llama']
    else:
        return row['Nemo']

# Apply the model logic to the test set
test_data['Model1_Prediction'] = test_data.apply(model_1, axis=1)
test_data['Model2_Prediction'] = test_data.apply(model_2, axis=1)

# Evaluate each model
def evaluate_model(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label='safe')
    recall = recall_score(true_labels, predictions, pos_label='safe')
    f1 = f1_score(true_labels, predictions, pos_label='safe')
    return accuracy, precision, recall, f1

model1_metrics = evaluate_model(test_data['Model1_Prediction'], test_data['True'])
model2_metrics = evaluate_model(test_data['Model2_Prediction'], test_data['True'])

print("Model 1 Metrics (Accuracy, Precision, Recall, F1):", model1_metrics)
print("Model 2 Metrics (Accuracy, Precision, Recall, F1):", model2_metrics)
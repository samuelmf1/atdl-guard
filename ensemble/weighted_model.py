import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reading the CSV files
df1 = pd.read_csv('llamaguard/results/guardrail_llama_test_results.csv')
df2 = pd.read_csv('nemo/guardrail_nemo_test_results.csv')

# Combining datasets with necessary columns and the prompt for reference
df_combined = pd.concat([
    df1[['predicted', 'expected', 'prompt']].rename(columns={'predicted': 'llama_prediction'}),
    df2['predicted'].rename('nemo_prediction')
], axis=1)

# Convert specific values in llama_prediction and nemo_predicted
df_combined['llama_prediction'] = df_combined['llama_prediction'].map({'unsafe': True, 'safe': False})
df_combined['nemo_prediction'] = df_combined['nemo_prediction'].map({'unsafe': True, 'safe': False})
df_combined['expected'] = df_combined['expected'].map({'unsafe': True, 'safe': False})

# Defining features and labels
X = df_combined[['llama_prediction', 'nemo_prediction']]
y = df_combined['expected']

# Including prompt for tracking in splits
X['prompt'] = df_combined['prompt']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate prompts for use in results
prompts_train = X_train['prompt']
prompts_test = X_test['prompt']

# Remove the prompt column from X_train and X_test for training and prediction
X_train = X_train.drop(columns=['prompt'])
X_test = X_test.drop(columns=['prompt'])

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
# print("Confusion Matrix:\n", confusion_matrix(y_test, predictions, labels=[True, False]))


# Preparing the results DataFrame
results_df = pd.DataFrame({
    'Prompt': prompts_test,
    'Predicted': predictions,
    'Actual': y_test
})

# Write the DataFrame to a CSV file
results_df.to_csv('ensemble/weighted_ensemble_predictions.csv', index=False)

# Filter to include only misclassified examples
failed_predictions_df = results_df[results_df['Predicted'] != results_df['Actual']]

# Write the misclassified predictions to a CSV file
failed_predictions_df.to_csv('ensemble/failed_ensemble_predictions.csv', index=False)

def custom_confusion_matrix(actual, predicted, positive_label):

    # Initialize the counters
    tp = tn = fp = fn = 0

    # Iterate over the actual and predicted labels
    for a, p in zip(actual, predicted):
        if a == positive_label and p == positive_label:
            tp += 1
        elif a == positive_label and p != positive_label:
            fn += 1
        elif a != positive_label and p == positive_label:
            fp += 1
        elif a != positive_label and p != positive_label:
            tn += 1

    # Print the results
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')

    return tp, fp, fn, tn

custom_confusion_matrix(y_test, predictions, True)
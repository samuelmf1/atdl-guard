import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Reading the CSV files
df1 = pd.read_csv('llamaguard/results/guardrail_llama_test_results.csv')
df2 = pd.read_csv('nemo/guardrail_nemo_test_results.csv')

# Combining datasets with necessary columns
df_combined = pd.concat([
    df1[['predicted', 'expected', 'prompt']].rename(columns={'predicted': 'llama_prediction'}),
    df2[['predicted']].rename(columns={'predicted': 'nemo_prediction'})
], axis=1)

# Map 'safe' and 'unsafe' to boolean
df_combined['llama_prediction'] = df_combined['llama_prediction'].map({'unsafe': True, 'safe': False})
df_combined['nemo_prediction'] = df_combined['nemo_prediction'].map({'unsafe': True, 'safe': False})
df_combined['expected'] = df_combined['expected'].map({'unsafe': True, 'safe': False})

# Extract prompts for use only in results reporting
prompts = df_combined['prompt']

# Defining features and labels without the prompt text embeddings
X = df_combined[['llama_prediction', 'nemo_prediction']]
y = df_combined['expected']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Keep prompts corresponding to the test data
test_prompts = prompts.iloc[X_test.index]

# Initialize and train the MLP Classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(2,), max_iter=100, activation='relu', solver='adam', random_state=42)
mlp_classifier.fit(X_train, y_train)

# Make predictions
predictions = mlp_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Preparing the results DataFrame
results_df = pd.DataFrame({
    'Prompt': test_prompts,
    'Predicted': predictions,
    'Actual': y_test
})

# Write the DataFrame to a CSV file
results_df.to_csv('ensemble/deep_ensemble_predictions_no_prompt.csv', index=False)

# Filter to include only misclassified examples
failed_predictions_df = results_df[results_df['Predicted'] != results_df['Actual']]
failed_predictions_df.to_csv('ensemble/deep_ensemble_predictions_no_prompt_failed.csv', index=False)

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

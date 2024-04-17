import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df1 = pd.read_csv('llamaguard/results/guardrail_llama_test_results.csv')
df2 = pd.read_csv('nemo/guardrail_nemo_test_results.csv')

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

# Evaluate each model
def evaluate_model(prompts, predictions, true_labels, model_name):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label='unsafe')
    recall = recall_score(true_labels, predictions, pos_label='unsafe')
    f1 = f1_score(true_labels, predictions, pos_label='unsafe')
    custom_confusion_matrix(true_labels, predictions, "unsafe")
    print(f"{model_name} Metrics (Accuracy, Precision, Recall, F1):", accuracy, precision, recall, f1)

    # Preparing the results DataFrame
    results_df = pd.DataFrame({
        'Prompt': prompts,
        'Predicted': predictions,
        'Actual': true_labels
    })

    # Write the DataFrame to a CSV file
    results_df.to_csv(f"ensemble/{model_name}_model_predictions.csv", index=False)

    # Filter to include only misclassified examples
    failed_predictions_df = results_df[results_df['Predicted'] != results_df['Actual']]

    # Write the misclassified predictions to a CSV file
    failed_predictions_df.to_csv(f"ensemble/failed_{model_name}_model_predictions.csv", index=False)

evaluate_model(df1['prompt'],  df1['predicted'], df1['expected'], "llama")
evaluate_model(df2['prompt'], df2['predicted'], df2['expected'], "nemo")
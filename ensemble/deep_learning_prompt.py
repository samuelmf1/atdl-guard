import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# Reading the CSV files
df1 = pd.read_csv('llamaguard/results/guardrail_llama_test_results.csv')
df2 = pd.read_csv('nemo/guardrail_nemo_test_results.csv')

# Combining datasets with necessary columns
df_combined = pd.concat([
    df1[['predicted', 'expected', 'prompt']].rename(columns={'predicted': 'llama_prediction'}),
    df2['predicted'].rename('nemo_prediction')
], axis=1)

# Map 'safe' and 'unsafe' to boolean
df_combined['llama_prediction'] = df_combined['llama_prediction'].map({'unsafe': True, 'safe': False})
df_combined['nemo_prediction'] = df_combined['nemo_prediction'].map({'unsafe': True, 'safe': False})
df_combined['expected'] = df_combined['expected'].map({'unsafe': True, 'safe': False})

# Prepare the text features using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(df_combined['prompt'])
X_text_df = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())

# Concatenate vectorized text with other features
X = pd.concat([df_combined[['llama_prediction', 'nemo_prediction']], X_text_df], axis=1)
y = df_combined['expected']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the MLP Classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(5,), max_iter=100, activation='relu', solver='adam', random_state=42)
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
    'Prompt': df_combined.loc[X_test.index, 'prompt'],
    'Predicted': predictions,
    'Actual': y_test
})

# Write the DataFrame to a CSV file
results_df.to_csv('ensemble/deep_ensemble_predictions_with_prompt.csv', index=False)

# Filter to include only misclassified examples
failed_predictions_df = results_df[results_df['Predicted'] != results_df['Actual']]

# Write the misclassified predictions to a CSV file
failed_predictions_df.to_csv('ensemble/deep_ensemble_predictions_with_prompt_failed.csv', index=False)

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


from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Apply PCA to reduce dimensions for visualization
pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))
X_train_pca = pca_pipeline.fit_transform(X_train)  # Fitting and transforming train data
X_test_pca = pca_pipeline.transform(X_test)  # Transforming test data

# Train MLP on reduced dimension data (for visualization purposes)
mlp_classifier.fit(X_train_pca, y_train)

# Function to plot decision boundaries after dimensionality reduction
def plot_decision_boundaries(X, y, model, title="Decision Boundary"):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Call the function to plot the decision boundaries
plot_decision_boundaries(X_test_pca, y_test, mlp_classifier, "MLP Decision Boundary on Test Data with PCA")

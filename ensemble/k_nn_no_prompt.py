import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

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

# Extract prompt for later reference
prompts = df_combined['prompt']

# Defining features and labels
X = df_combined[['llama_prediction', 'nemo_prediction']]
y = df_combined['expected']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Keep prompts corresponding to the test data
test_prompts = prompts.iloc[X_test.index]

# Initialize and train the kNN Classifier with a grid search
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
knn_classifier = KNeighborsClassifier()
grid = GridSearchCV(knn_classifier, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train, y_train)

# Using the best parameters to make predictions
best_knn = grid.best_estimator_
predictions = best_knn.predict(X_test)

# Best parameters and best score
print("Best Score:", grid.best_score_)
print("Best Parameters:", grid.best_params_)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Preparing the results DataFrame for misclassified examples
failed_predictions_df = pd.DataFrame({
    'Prompt': test_prompts[y_test != predictions].values,
    'Predicted': predictions[y_test != predictions],
    'Actual': y_test[y_test != predictions]
})

# Write the misclassified predictions to a CSV file
failed_predictions_df.to_csv('ensemble/knn_ensemble_failed_predictions.csv', index=False)

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

def plot_decision_boundaries(X, y, model, title="Decision Boundary"):
    h = 0.01  # step size in the mesh
    # Create color maps
    cmap_light = plt.cm.RdYlBu
    cmap_bold = plt.cm.Dark2

    # Calculate min, max and limits
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1

    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h),
                         np.arange(y_min, y_max + h, h))

    # Predict classifications for each point in mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot the training points
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50, marker='o')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('LlamaGuard Prediction')
    plt.ylabel('NeMo Prediction')

    # Create a legend from custom artist/label lists
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper left", title="Classes")
    plt.gca().add_artist(legend1)

    plt.grid(True)
    plt.show()

# Plot the decision boundary using the training data
plot_decision_boundaries(X_train, y_train, best_knn, title="kNN Decision Boundary")

import pandas as pd

def process_file(filepath, x):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Initialize lists to store data
    safe_data = []
    unsafe_data = []

    # Read every four lines as one record
    for i in range(0, len(lines), 4):
        try:
            prompt = lines[i].strip()
            label = lines[i + 1].strip().lower()
            explanation = lines[i + 2].strip()

            if label == 'safe':
                safe_data.append([prompt, label, explanation])
            elif label == 'unsafe':
                unsafe_data.append([prompt, label, explanation])
        except IndexError:
            break  # In case the file doesn't end perfectly

    # Convert lists to DataFrames
    df_safe = pd.DataFrame(safe_data, columns=['Prompt', 'Label', 'Explanation'])
    df_unsafe = pd.DataFrame(unsafe_data, columns=['Prompt', 'Label', 'Explanation'])

    # Sample x rows from each DataFrame
    df_safe_sample = df_safe.sample(n=x, random_state=42)
    df_unsafe_sample = df_unsafe.sample(n=x, random_state=42)

    # Combine the samples
    combined_sample = pd.concat([df_safe_sample, df_unsafe_sample])

    # Save the combined sample to a CSV file
    combined_sample.to_csv('sampled_data.csv', index=False)

# Example usage
process_file('data.txt', 5) 
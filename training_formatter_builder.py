class TrainingExample:
    def __init__(self, prompt, response, violated_category_codes, label, explanation):
        self.prompt = prompt
        self.response = response
        self.violated_category_codes = violated_category_codes
        self.label = label
        self.explanation = explanation

def parse_text_file(file_path):
    examples = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]
        for i in range(0, len(lines), 4):
            if i + 1 < len(lines):
                prompt = lines[i].strip()
                label = lines[i + 1].strip()
            else:
                break  # Break the loop if there are not enough lines left
            if i + 2 < len(lines):
                explanation = lines[i + 2].strip()
            else:
                explanation = ""

            if label == "safe":
                response = "N/A"
                violated_category_codes = []
            else:
                response = "N/A"  # Update with response later if needed
                violated_category_codes = ["O7"]

            count += 1
            example = TrainingExample(
                prompt=prompt,
                response=response,
                violated_category_codes=violated_category_codes,
                label=label,
                explanation=explanation
            )
            examples.append(example)
    return examples

def format_example_as_code(example):
    return f"TrainingExample(\n\tprompt=\"{example.prompt}\",\n\tresponse=\"{example.response}\",\n\tviolated_category_codes={example.violated_category_codes},\n\tlabel=\"{example.label}\",\n\texplanation=\"{example.explanation}\"\n),"

def main():
    file_path = './soda_training.txt'  # path to text file
    examples = parse_text_file(file_path)
    
    for example in examples:
        print(format_example_as_code(example))

if __name__ == "__main__":
    main()

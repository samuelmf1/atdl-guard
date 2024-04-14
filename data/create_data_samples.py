from openai import OpenAI

safe_example = """ 
Who are the key executives of A Soda Company?
safe
The prompt asks about individuals associated with only 'A Soda Company'.

What are the sustainability practices of A Soda Company?
safe
The prompt asks about environmental initiatives of specifically 'A Soda Company'.

"""

unsafe_example = """
How has 'A Soda Company' been affected by the recent regulatory changes, and how does it differ from Coca-Cola?
unsafe
The prompt includes discussing the impact of regulatory changes on 'The Soda Company' in comparison to another company.

Please provide insights on the customer demographics for 'The Soda Company' and Amazon.
unsafe
The prompt is asking about customer demographics for 'The Soda Company' compared to another company.

What investments has Sierra Mist made in expanding its operations and how can A Soda Company replicate this success?
unsafe
The prompt discusses investments made by another company.

What flavors does Bubble Up have?
unsafe
The prompt is asking about information regarding a company that is not 'A Soda Company'.

"""

def generateExamples(safeVar):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an assistant that will help me create mock data to train my LLM guardrails"},
        {"role": "user", "content": "I need you to help me generate data for an LLM, specifically llama guard by Meta. I specially am trying to create guardrails that prevent user prompts from asking information that pertains to other companies. In specific, I am pretending I am a soda company, named 'A Soda Company', and I want to block all prompts that ask anything related to other companies other than 'A Soda Company'"}, 
        {"role": "user", "content": f"""User
        In specific, I will create 25 instances for the initial training set. I want your help to do it. I specifically need instances that are "safe" and "unsafe" where safe meant it did not violate this. Here is an example of how llamaguard takes in training data: training_examples = [
        TrainingExample(
            prompt="What is the home address of John Smith?",
            label="unsafe",
            explanation="The prompt asks for a home address."
        ) 
        ] . As you can see I need a print, label, and explanation. Imagine my company is called "The Soda Company". In terms of what is considered safe and unsafe, I will reiterate once more. A prompt in this case is considered unsafe if the prompt inquires about any company other than 'A Soda Company', such as 
         asking about comparison of A Soda Compnay to other companies. It is considered safe if the prompt only pertains to 'A Soda Company', such as asking about any information about 'A Soda Company'. 
          Can you give me 25 training examples which include prompt, label, and explanation (and make sure the explanation is very simple like the example I showed you). 
         I specifically want ALL 25 of these training examples to be {safeVar}. For simplicity for each training example have 3 lines where the first line is the prompt, second is label, and third is explanation. 
         And have exactly one newline between examples. Just give me these examples in your response, don't put any other text in the response and dont number the lines. And make sure after the final example you put exactly one newline. Thus, the output should have exactly 100 lines (3 lines per example and onenewline between each example including after last example)

        Here is how an example of explicitly how I want the format: {safe_example if safeVar == "safe" else unsafe_example}
    """}
    ]
    )

    output_file_path = 'full_sample.txt'
    content = completion.choices[0].message.content
    with open(output_file_path, 'a') as file:
        file.write(content)


client = OpenAI()
safeVar = True

# make sure this is an even number as it switches between safe and unsafe generation 
# generates 25 per batch, so 40 will generate 1000 examples with half safe/unsafe
num_examples = 40
for i in range(num_examples): 
    generateExamples("safe" if safeVar else "unsafe")
    safeVar = not safeVar
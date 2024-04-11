from nemoguardrails import LLMRails, RailsConfig
import os
import pandas as pd
os.environ["OPENAI_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = RailsConfig.from_path("./config")
rails = LLMRails(config)


prompts=[]
labels = []
with open("../test_samples.txt") as file:
    for i,item in enumerate(file):        
        if (i+1)%2==0 and not (i+1)%4==0:

            prompts.append(item.replace('\n', ''))
        if (i+1)%2!=0 and item!="\n":
            labels.append(item.replace('\n', ''))
false = 0
correct = 0

####### You can decrease the number of prompts below so we dont waste tokens in testing
# labels = labels[0:2]
# prompts = prompts[0:2]

total = len(prompts)

df = pd.DataFrame(columns=['Prompt', 'True', 'Predicted'])

for label,prompt in zip(labels,prompts):
    try:
        res = rails.generate(prompt=prompt,return_context=True)
        pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        res[1]['_last_bot_prompt']
        if label == "unsafe":
            false+=1
        else:
            correct +=1
        row = {'Prompt': prompt, 'True': label, 'Predicted': 'safe'}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    except:
        if label=="unsafe":
            correct +=1
        else:
            false +=1
        row = {'Prompt': prompt, 'True': label, 'Predicted': 'unsafe'}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

df.to_csv("predictions.csv",index=False)
accuracy = correct/total
print(accuracy)
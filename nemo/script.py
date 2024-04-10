from nemoguardrails import LLMRails, RailsConfig
import os
os.environ["OPENAI_API_KEY"] = "key_heres"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = RailsConfig.from_path("./config")
rails = LLMRails(config)


prompts=[]
labels = []
with open("../test_samples.txt") as file:
    for i,item in enumerate(file):        
        if (i+1)%2==0 and not (i+1)%4==0:

            prompts.append(item)
        if (i+1)%2!=0 and item!="\n":
            labels.append(item)
false = 0
total = len(prompts)
correct = 0

labels = labels[0:1]
prompts = prompts[0:1]
for label,prompt in zip(labels,prompts):
    try:
        res = rails.generate(prompt="Tell me about the Soda company",return_context=True)
        res[1]['_last_bot_prompt']
        if label == "unsafe\n":
            false+=1
        else:
            correct +=1
    except:
        if label=="unsafe\n":
            correct +=1
        else:
            false +=1
accuracy = correct/total
print(accuracy)
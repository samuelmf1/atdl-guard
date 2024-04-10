from nemoguardrails import LLMRails, RailsConfig
import os
os.environ["OPENAI_API_KEY"] = "openai_key here"
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

#define prompts
#define labels
false = 0
total = len(prompts)
correct = 0
for label,prompt in zip(labels,prompts):

    try:
        res = await rails.generate_async(prompt="Tell me about the Soda company")
        res[1]['_last_bot_prompt']
        if label == "unsafe":
            false+=1
        else:
            correct +=1
    except:
        if label=="unsafe":
            correct +=1
        else:
            false +=1
accuracy = correct/total
print(accuracy)
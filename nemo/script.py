from nemoguardrails import LLMRails, RailsConfig
import os
os.environ["OPENAI_API_KEY"] = "openai_key here"
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

#define prompts
false = 0
total = len(prompts)
for prompt in prompts:

    try:
        res = await rails.generate_async(prompt="Tell me about the Soda company")
        res[1]['_last_bot_prompt']
    except:
        false +=1

accuracy = (total - false)/false
print(accuracy)
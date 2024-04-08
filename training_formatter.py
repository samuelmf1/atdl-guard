# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from finetuning_data_formatter import TrainingExample, Guidelines, Category, LlamaGuardPromptConfigs, LlamaGuardGenerationConfigs, ExplanationPosition, AugmentationConfigs, FormatterConfigs, create_formatted_finetuning_examples

# Define a list of guidelines and training examples to match
guidelines = Guidelines(
    categories=[
        Category(name="Competitors",
                 description="Information about a competitor company.")
    ],
    category_code_prefix="O"
)

training_examples = [
    TrainingExample(
        prompt="What are the main ingredients in The Soda Company's flagship soda?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks for information about our product."
    ),
    TrainingExample(
        prompt="Has The Soda Company ever collaborated with DrinkWell Corp?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about partnerships with another company."
    ),
    TrainingExample(
        prompt="Can you list all the countries where The Soda Company products are available?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt requests information about our product distribution."
    ),
    TrainingExample(
        prompt="What's the employee count of The Soda Company compared to QuenchCo?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks for comparative information with another company."
    ),
    TrainingExample(
        prompt="Describe the process The Soda Company uses to carbonate its beverages.",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks for details about our production process."
    ),
    TrainingExample(
        prompt="Which company has a larger product line, The Soda Company or FizzJoy?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests a product line comparison with a competitor."
    ),
    TrainingExample(
        prompt="What initiatives has The Soda Company taken to support local communities?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our community support initiatives."
    ),
    TrainingExample(
        prompt="Detail the market strategy The Soda Company uses against SipQuick.",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks strategic information against a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company ensure its packaging is eco-friendly?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our environmental efforts."
    ),
    TrainingExample(
        prompt="What's the pricing strategy of The Soda Company versus Refreshz?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks for a comparison of pricing strategies with a competitor."
    ),
    TrainingExample(
        prompt="What new flavors is The Soda Company planning to launch this year?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about upcoming product launches."
    ),
    TrainingExample(
        prompt="How do sales figures for The Soda Company compare to those of BubblyPop last quarter?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks sales comparisons with a competitor."
    ),
    TrainingExample(
        prompt="What awards for sustainability has The Soda Company received?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about awards related to sustainability."
    ),
    TrainingExample(
        prompt="Can you provide a SWOT analysis for The Soda Company and its main competitors?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests strategic analysis involving competitors."
    ),
    TrainingExample(
        prompt="How does The Soda Company source its natural flavors?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about the sourcing of ingredients."
    ),
    TrainingExample(
        prompt="What is The Soda Company's stance on sugar alternatives?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks for information on our product ingredients."
    ),
    TrainingExample(
        prompt="Does The Soda Company have a recycling program for its products?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our environmental initiatives."
    ),
    TrainingExample(
        prompt="How often does The Soda Company release new flavors?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our product development cycle."
    ),
    TrainingExample(
        prompt="Are there any job openings at The Soda Company in marketing?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about employment opportunities."
    ),
    TrainingExample(
        prompt="What is the calorie count for The Soda Company's most popular soda?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks for nutritional information about our products."
    ),
    TrainingExample(
        prompt="Can The Soda Company sodas be found in eco-friendly packaging?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about packaging sustainability."
    ),
    TrainingExample(
        prompt="What was The Soda Company's founding year?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks for historical information about the company."
    ),
    TrainingExample(
        prompt="How does The Soda Company support local farmers?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our supply chain practices."
    ),
    TrainingExample(
        prompt="What are the top-selling products of The Soda Company?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks for information on our popular products."
    ),
    TrainingExample(
        prompt="How does The Soda Company engage with its community on social media?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our social media engagement."
    ),
    TrainingExample(
        prompt="Is The Soda Company planning to expand to new markets?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our market expansion strategies."
    ),
    TrainingExample(
        prompt="What corporate social responsibility initiatives does The Soda Company have?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our CSR initiatives."
    ),
    TrainingExample(
        prompt="How does The Soda Company ensure product safety and quality?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our quality assurance processes."
    ),
    TrainingExample(
        prompt="What kind of sponsorships does The Soda Company participate in?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our sponsorship activities."
    ),
    TrainingExample(
        prompt="Does The Soda Company have any zero-sugar options?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our product range regarding sugar content."
    ),
    TrainingExample(
        prompt="What technology does The Soda Company use in production?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our production technology."
    ),
    TrainingExample(
        prompt="How is The Soda Company addressing climate change?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our environmental impact and initiatives."
    ),
    TrainingExample(
        prompt="What are the challenges faced by The Soda Company in the beverage industry?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt may lead to discussion of industry-wide issues, including those faced by competitors."
    ),
    TrainingExample(
        prompt="How does The Soda Company's research and development process work?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our R&D processes."
    ),
    TrainingExample(
        prompt="What are The Soda Company's future plans for sustainability?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about future environmental initiatives."
    ),
    TrainingExample(
        prompt="Can you compare The Soda Company's ethical sourcing practices with those of other beverage companies?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests a comparison with other companies' practices."
    ),
    TrainingExample(
        prompt="What awards has The Soda Company received for innovation?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our recognition in innovation."
    ),
    TrainingExample(
        prompt="How has The Soda Company's market strategy evolved over the last decade?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our strategic evolution."
    ),
    TrainingExample(
        prompt="What makes The Soda Company's beverages different from homemade sodas?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks for information about the unique qualities of our products."
    ),
    TrainingExample(
        prompt="Does The Soda Company offer tours of its manufacturing facilities?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our public engagement initiatives."
    ),
    TrainingExample(
        prompt="What health and wellness initiatives does The Soda Company promote?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our involvement in health and wellness."
    ),
    TrainingExample(
        prompt="How does The Soda Company choose its flavors?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our flavor selection process."
    ),
    TrainingExample(
        prompt="What is The Soda Company's policy on artificial sweeteners?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks for our stance on ingredient usage."
    ),
    TrainingExample(
        prompt="How does The Soda Company contribute to the local economy?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our economic impact."
    ),
    TrainingExample(
        prompt="Does The Soda Company have a loyalty program?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about customer engagement strategies."
    ),
    TrainingExample(
        prompt="What are The Soda Company's goals for the next five years?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our future objectives."
    ),
    TrainingExample(
        prompt="How does The Soda Company manage its water usage?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our environmental conservation efforts."
    ),
    TrainingExample(
        prompt="What partnerships has The Soda Company formed to enhance its sustainability efforts?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our collaborative environmental initiatives."
    ),
    TrainingExample(
        prompt="How are The Soda Company's beverages distributed globally?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our distribution strategies."
    ),
    TrainingExample(
        prompt="Is The Soda Company involved in any educational initiatives?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt asks about our involvement in education."
    ),
    TrainingExample(
        prompt="How does The Soda Company handle feedback and complaints?",
        response="N/A",
        violated_category_codes=[],
        label="safe",
        explanation="The prompt inquires about our customer service processes."
    ),
    TrainingExample(
        prompt="Does The Soda Company use the same suppliers as FizzyPop?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks for comparison on suppliers with a competitor."
    ),
    TrainingExample(
        prompt="What market share does The Soda Company hold compared to SparkleDrink?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests market share comparison with a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company's advertising budget compare to that of FreshFizz?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks comparison of advertising budgets with a competitor."
    ),
    TrainingExample(
        prompt="What are the differences in employee benefits between The Soda Company and BubbleBlast?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about employee benefits in comparison to those of a competitor."
    ),
    TrainingExample(
        prompt="Can you provide sales data for The Soda Company vs. PureSpritz?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests sales data in comparison with a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company's environmental record stack up against GreenGuzzle?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks for environmental performance comparison with a competitor."
    ),
    TrainingExample(
        prompt="What strategic advantages does The Soda Company have over SodaSplash?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks information on strategic advantages over a competitor."
    ),
    TrainingExample(
        prompt="How do the flavors of The Soda Company compare to those of EcoFizz?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks for a product comparison with a competitor."
    ),
    TrainingExample(
        prompt="What initiatives has The Soda Company taken to outperform QuenchQuick in the market?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about competitive strategies against a specific competitor."
    ),
    TrainingExample(
        prompt="Has The Soda Company ever considered a merger with FizzFast?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks about potential mergers with a competitor."
    ),
    TrainingExample(
        prompt="How do The Soda Company's product prices compare with those of VitaPop?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests price comparisons with a competitor."
    ),
    TrainingExample(
        prompt="What is the public perception of The Soda Company compared to ThirstQuencher?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks comparison of public perception with a competitor."
    ),
    TrainingExample(
        prompt="Why do consumers prefer The Soda Company over BuzzBeverages?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires why consumers might prefer our company over a competitor."
    ),
    TrainingExample(
        prompt="What legal disputes has The Soda Company had with SipNSparkle?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks about legal disputes with a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company's social responsibility efforts compare to HydrateHype's?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests comparison of social responsibility efforts with a competitor."
    ),
    TrainingExample(
        prompt="What are the patent disputes between The Soda Company and LiquidLuxe?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about specific patent disputes with a competitor."
    ),
    TrainingExample(
        prompt="How has The Soda Company responded to the marketing tactics of RefreshRush?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks information on response to competitor marketing tactics."
    ),
    TrainingExample(
        prompt="What are the unique selling points of The Soda Company against GlobeSip?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks for USPs in comparison to a competitor."
    ),
    TrainingExample(
        prompt="How do The Soda Company's sustainability practices differ from those of PlanetPop?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests comparison of sustainability practices with a competitor."
    ),
    TrainingExample(
        prompt="Why is The Soda Company more successful in market penetration than FizzFlare?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about market penetration success over a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company's R&D investment compare to that of InnovateIndulge?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks comparison of R&D investment with a competitor."
    ),
    TrainingExample(
        prompt="What makes The Soda Company a better employer than DrinkDream?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks why our company is a better employer than a competitor."
    ),
    TrainingExample(
        prompt="Has The Soda Company won more industry awards than FlavorFountain?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about industry awards in comparison to a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company plan to compete with the new products of MixMagic?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks strategy against competitor's new products."
    ),
    TrainingExample(
        prompt="What were The Soda Company's sales figures compared to those of ThriveThirst last year?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests last year's sales figures in comparison with a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company's customer loyalty program compare to SizzleSip?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks for comparison of loyalty programs with a competitor."
    ),
    TrainingExample(
        prompt="Why did The Soda Company's recent campaign outperform RefreshRevolution's?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about campaign performance over a competitor."
    ),
    TrainingExample(
        prompt="What market segments does The Soda Company target differently than DewDrop?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks information on target market segments compared to a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company's carbon footprint compare to EcoSoda?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests comparison of carbon footprints with a competitor."
    ),
    TrainingExample(
        prompt="What technological innovations set The Soda Company apart from TechTaste?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks about technological innovations compared to a competitor."
    ),
    TrainingExample(
        prompt="How are The Soda Company's logistics and distribution more efficient than QuickQuench?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks efficiency comparison in logistics with a competitor."
    ),
    TrainingExample(
        prompt="What consumer demographics prefer The Soda Company over PurelyPop?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about consumer demographics preference over a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company's global reach compare to WorldWideWet?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests comparison of global reach with a competitor."
    ),
    TrainingExample(
        prompt="What collaborations has The Soda Company engaged in to outdo SplashSensation?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks information on collaborations to outdo a competitor."
    ),
    TrainingExample(
        prompt="How do The Soda Company's packaging innovations compare to NatureNectar?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks for comparison of packaging innovations with a competitor."
    ),
    TrainingExample(
        prompt="What challenges has The Soda Company faced in competing with AquaAwe?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about challenges in competing with a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company's growth rate compare to RapidRefresh?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests comparison of growth rates with a competitor."
    ),
    TrainingExample(
        prompt="What tactics has The Soda Company used to gain market share from ZestZoom?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks information on tactics used against a competitor."
    ),
    TrainingExample(
        prompt="Why does The Soda Company have a higher customer satisfaction rating than BlissBubbles?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about higher customer satisfaction than a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company's brand value compare to EliteFizz?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests comparison of brand values with a competitor."
    ),
    TrainingExample(
        prompt="What distribution channels does The Soda Company leverage better than BounceBeverage?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt asks about better utilization of distribution channels than a competitor."
    ),
    TrainingExample(
        prompt="How has The Soda Company's innovation strategy differed from CreativityCola?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt seeks differences in innovation strategy from a competitor."
    ),
    TrainingExample(
        prompt="What product recalls has The Soda Company had compared to SafeSip?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt inquires about product recalls compared to a competitor."
    ),
    TrainingExample(
        prompt="How does The Soda Company maintain higher product standards than PrimePotion?",
        response="N/A",
        violated_category_codes=['O7'],
        label="unsafe",
        explanation="The prompt requests information on maintaining higher standards than a competitor."
    )
]

# Setting up configs
llama_guard_prompt_configs = LlamaGuardPromptConfigs(
    instructions_format_string="""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{guidelines}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{conversation}

<END CONVERSATION>

Provide your safety assessment for $agent_type in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""",
    should_include_category_descriptions=True,
    should_shuffle_category_codes=True
)

llama_guard_generation_configs = LlamaGuardGenerationConfigs(
    should_list_violated_codes=True,
    explanation_position=ExplanationPosition.AFTER_DECISION
)

augmentation_configs = AugmentationConfigs(
    should_add_examples_with_dropped_nonviolated_prompt_categories=True,
    should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=True,
    explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect."
)

formatter_configs = FormatterConfigs(
    guidelines=guidelines,
    llama_guard_prompt_configs=llama_guard_prompt_configs,
    llama_guard_generation_configs=llama_guard_generation_configs,
    augmentation_configs=augmentation_configs,
    random_seed=42
)

# Call the create_formatted_finetuning_examples function
formatted_examples = create_formatted_finetuning_examples(
    training_examples, formatter_configs)

# Print the formatted examples
print(formatted_examples)

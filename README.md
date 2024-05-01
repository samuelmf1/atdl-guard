# Enhancing LLM Guardrails: A Comparative Analysis Using Ensemble Techniques

Anuva Banwasi, Samuel Friedman, Michael Khanzadeh, Harinder Mashiana

COMS 6998: Advanced Topics in Deep Learning (ATDL), Spring 2024

Columbia University, Departments of Computer Science and Data Science

## Abstract

We present an approach for enhancing LLM guardrails through a comparative analysis using ensemble techniques. We examine two main approaches, Llama Guard and NeMo guardrails, which represent LLM-based and vector similarity search approaches, respectively. Our study aims to explore the effectiveness of these frameworks in practical scenarios, emphasizing the need for robust content moderation interfaces between users and LLMs.

We conduct a comparative study between Llama Guard and NeMo, evaluating their accuracy on new guardrail categories. We then propose novel strategies to improve upon existing guardrails, including fine-tuning Llama Guard for novel guardrail categories with practical use cases. Experimental results reveal the impact of dataset size and model configuration on the effectiveness of guardrail enforcement.

Overall, this research contributes to the advancement of content moderation paradigms for LLMs, providing insights into the potential for ensemble techniques to enhance guardrail enforcement in real-world applications.

## Documentation

The repository contains all documentation, including our [Enhancing LLM Guardrails paper](Enhancing_LLM_Guardrails_Paper.pdf), [slides](ATDL_Guard_Slides.pdf), and code.

- [/data](/data): our custom training data
- [/llamaguard](/llamaguard): training and fine tuning Llama Guard model using our data and use case
- [/nemo](/nemo): our experimentation with NeMo Guard using our data and use case
- [/rephrase_eval](/rephrase_eval): study how well NeMo can capture similar semantic representations of a single config prompt
- [/ensemble](/ensemble): applying ensemble methods to integrate models in attempt to increase overall accuracy

Our PEFT Llama Guard model, which we trained using our custom dataset, can be found at [huggingface.co/anuvab/llama-guard-finetuned-1ep-1000](https://huggingface.co/anuvab/llama-guard-finetuned-1ep-1000).

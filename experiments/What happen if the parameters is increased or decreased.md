## max\_new\_tokens

* Increasing: Setting a higher value (e.g., 1000\) allows the model to generate longer outputs, which can be useful for detailed responses or narratives. However, it may also lead to less coherent outputs if the context is not maintained.  
* Decreasing: A lower value restricts the output length, resulting in shorter and potentially more focused responses. This can help maintain coherence but may limit the depth of information provided.

## temperature

* Increasing: Raising the temperature above 0.5 (e.g., to 1.0 or higher) increases randomness in the output, making it more creative and diverse. However, this can also lead to less coherent or relevant responses as the model explores less likely options[1](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)[2](https://community.openai.com/t/a-better-explanation-of-top-p/2426)[3](https://becomingahacker.org/understanding-key-ai-language-model-parameters-top-p-temperature-num-beams-and-do-sample-9874bf3c89ae?gi=d27378ac63a6).  
* Decreasing: Lowering the temperature (e.g., to 0.2 or 0.3) makes the model more deterministic, favoring more probable tokens. This results in safer and more predictable outputs, which is often desirable for tasks requiring accuracy[1](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)[3](https://becomingahacker.org/understanding-key-ai-language-model-parameters-top-p-temperature-num-beams-and-do-sample-9874bf3c89ae?gi=d27378ac63a6).

## top\_p

* Increasing: Setting a higher top\_p (closer to 1\) allows a broader selection of possible tokens, which can enhance creativity but may also introduce irrelevant or off-topic content. It effectively expands the pool of choices available to the model[2](https://community.openai.com/t/a-better-explanation-of-top-p/2426)[4](https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542).  
* Decreasing: A lower top\_p (e.g., 0.5) restricts the choice to only the most likely tokens, which can help maintain focus and relevance in responses but may reduce diversity and creativity[2](https://community.openai.com/t/a-better-explanation-of-top-p/2426)[4](https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542).

## top\_k

* Increasing: A higher top\_k value allows for sampling from a larger number of top tokens, which can increase variability and creativity in responses. However, it might also lead to less coherent outputs if too many options are considered[4](https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542).  
* Decreasing: Lowering top\_k narrows down the selection to only a few high-probability tokens, resulting in more predictable and focused outputs that are typically more relevant but less diverse[4](https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542).

## do\_sample

* Setting this to True: Enables sampling during generation rather than always selecting the most probable token. This increases variability and creativity in outputs.  
* Setting this to False: Forces the model to always choose the highest probability token, leading to more deterministic and predictable responses.

Optimal values:

| Parameter | Optimal Value |
| ----- | ----- |
| max\_new\_tokens | 512 \- 1024 |
| temperature | 0.2 \- 0.7 |
| top\_p | 0.9 \- 0.95 |
| top\_k | 40 \- 100 |
| repetition\_penalty | 1.1 \- 1.5 |
| do\_sample | True |


## **Prompt-and-Rerank**
### A Method for Zero-Shot and Few-Shot Arbitrary Textual Style Transfer with Small Language Models
##### — Authors: Mirac Suzgun, Luke Melas-Kyriazi, Dan Jurafsky
* [Paper Link](https://arxiv.org/abs/2205.11503) 
* [Website](https://lukemelas.github.io/prompt-and-rerank/)
* [Colab Notebook](https://colab.research.google.com/drive/1kpO7-KHOv39_T58Oi0Z2fKsAJ6WXK0YY?usp=sharing)

The following is an illustration of our proposed **Prompt-and-Rerank** method. Given an input text and the style transformation, we first compose a prompt and feed it to a pretrained language model (e.g., GPT-2) to generate multiple output texts—conditioned on the prompt—using beam search. We then re-score each candidate output along three axes, namely textual similarity, style transfer strength, and fluency. We choose the candidate with the highest re-ranked score as our output.

![Prompt-and-Rerank](https://github.com/suzgunmirac/prompt-and-rerank/blob/master/figures/PromptRerank.png)

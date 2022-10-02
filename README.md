## **Prompt-and-Rerank**
#### A Method for Zero-Shot and Few-Shot Arbitrary Textual Style Transfer with Small Language Models
##### — Authors: Mirac Suzgun, Luke Melas-Kyriazi, Dan Jurafsky
* [Paper Link](https://arxiv.org/abs/2205.11503) 
* [Website](https://lukemelas.github.io/prompt-and-rerank/)
* [Colab Notebook](https://colab.research.google.com/drive/1kpO7-KHOv39_T58Oi0Z2fKsAJ6WXK0YY?usp=sharing)

The following is an illustration of our proposed **Prompt-and-Rerank** method. Given an input text and the style transformation, we first compose a prompt and feed it to a pretrained language model (e.g., GPT-2) to generate multiple output texts—conditioned on the prompt—using beam search. We then re-score each candidate output along three axes, namely textual similarity, style transfer strength, and fluency. We choose the candidate with the highest re-ranked score as our output.

![Prompt-and-Rerank](https://github.com/suzgunmirac/prompt-and-rerank/blob/master/figures/PromptRerankViz.png)

**todo**:
- [ ] Upload the classifier models to Google Drive.
- [ ] Provide example commands for running inference code and evaluating the results.
- [ ] Upload the results.

### Running Inference
The following command uses GPT-2 Large to change the sentiment of Yelp restaurant review sentences from positive to negative and vice versa using the contrastive prompting method. It also uses curly brackets as delimiters (sentence boundary markers) and four prompt examplars, generates three outputs using beam search, and saves the results under the `outputs/yelp_clean/contrastive/gpt2_large/4_shot_3_samples` directory.

```bash
python run_inference.py \
    --model gpt2-large \
    --dataset yelp --clean_data \
    --delimiter curly_bracket --setting contrastive \
    --k_samples 3 --num_examplars 4 \
    --results_dir outputs/yelp_clean/contrastive/gpt2_large/4_shot_3_samples
```

### Evaluating Results
```bash
python calculate_all.py \
    --dataset yelp_clean \
    --output_dir outputs/yelp_clean/contrastive/gptj_6B/4_shot_3_samples \
    --choose_first \
    --results_save_path results/test.tsv
```

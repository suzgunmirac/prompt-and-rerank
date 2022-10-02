# Import relevant libraries and dependencies
import os
import json
import glob
import argparse
from tqdm import tqdm

import csv

from datasets import load_metric

import numpy as np
import torch
from torch.nn import functional as F
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

import math

from transformers import BertTokenizerFast

import datasets

FROM_TO_DICT={
    'positive': 'negative',
    'negative': 'positive',
    'formal': 'informal',
    'informal': 'formal',
    'ungrammatical': 'grammatical',
    'grammatical': 'ungrammatical',
    'Shakespearean': 'modern',
    'modern': 'Shakespearean',
    'oldEnglish': 'modernEnglish',
    'modernEnglish': 'oldEnglish',
    'symbolic': 'English',
    'English': 'symbolic',
}

def label_fn(model, label):
    return model.task.label_dictionary.string(
        [label + model.task.target_dictionary.nspecial]
    )


def compute_sentence_probs(model, tokenizer, preds, device):
    """
        Computes the sentence-level probability of texts in a list.
    """
    probs = []
    for pred in preds:
        sentence = pred
        sent_prob = get_sent_prob(model, tokenizer, sentence, device)
        probs.append(sent_prob)
    return probs


def get_sent_prob(model, tokenizer, text, device):
    """
        Calculates the sentence-level probability.
    """
    with torch.no_grad():
        text_ids = tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)
        if text_ids.shape[1] > 1:
            input_ids = text_ids
            target_ids = text_ids 
            outputs = model(input_ids, labels=target_ids)
            logits = outputs.logits
            probs = F.softmax(logits, dim=2).squeeze()
            sentence_prob = 1.
            
            for i in range(input_ids.shape[1]-1):
                current_probs = probs[i]
                gt_index = input_ids[0][i+1].cpu().item()
                gt_prob = current_probs[gt_index]
                sentence_prob *= gt_prob.cpu().item()
            return sentence_prob
        return 0.


def calc_sentiment_accuracy(model, tokenizer, predictions, target_style, device):
    """
        Given a model, a tokenizer, and a list of predictions, calculates the style-transfer strength (accuracy score) of the list.
    """
    if model is None:
        return None, 0., None
    with torch.no_grad():
        # first negative, second positive
        correcto =  np.zeros(2)
        correct = 0
        probs = []
        pbar = tqdm(predictions, desc='Calculating the style-transfer (sentiment) accuracy...')
        for i, sample in enumerate(pbar):
            # if Amazon or Yelp:
            if tokenizer:
                # If there is a tokenizer, then it means that we are doing sentiment classification and using BERT-classifiers
                input = tokenizer.encode(sample, return_tensors="pt", truncation=True, max_length=512).to(device)
                pred = model(input)
                positive_prob = torch.softmax(pred['logits'], dim=1)[0][1]
                if target_style == 'negative' or target_style == 'formal':
                    probs.append(1. - positive_prob.cpu())
                    if (1. - positive_prob) > 0.5:
                        correcto[0] += 1
                        correct += 1
                else:
                    probs.append(positive_prob.cpu())
                    if positive_prob > 0.5:
                        correcto[1] += 1
                        correct += 1
            else:
                # This means that we are using the Shakespeare classifier from https://drive.google.com/drive/folders/1XRVeNjxlXojEAzJOOyHVPg0z_9VyIpmo
                samples = [sample]

                inputs = [model.bpe.encode(sample) for sample in samples]
                input_batch = collate_tokens([model.task.source_dictionary.encode_line("<s> " + input + " </s>", append_eos=False) for input in inputs], pad_idx=1)
                input_batch = input_batch[:, : 512]

                with torch.no_grad():
                    preds = model.predict('sentence_classification_head', input_batch.long())
                pred_labels_label = [label_fn(model, x.argmax(axis=0).item()) for x in preds]
                if target_style == 'modern':
                    if pred_labels_label[0] == 'modern':
                        correcto[0] += 1
                        correct += 1
                else:
                    if pred_labels_label[0] == 'original':
                        correcto[1] += 1
                        correct += 1
        return correcto/float(len(predictions)), correct/float(len(predictions)), probs


def calc_blue(metric, predictions, references, lowercaseBool=True):
    """
        Given predictions and references, calculates the BLEU or sacreBLEU score.
    """
    results = metric.compute(predictions=predictions, references=references) #lowercase=lowercaseBool
    return results


def calc_perplexity_sent_average(model, tokenizer, predictions, device):
    """
        Calculates the sentence-level perplexity.
    """
    with torch.no_grad():
        ppls = []
        error_s = 0
        pbar = tqdm(predictions, desc='Calculating the sentence-level perplexity scores...')
        for i, sample in enumerate(pbar):
            tokens_tensor = tokenizer.encode(sample, return_tensors="pt", truncation=True, max_length=512).to(device)
            if tokens_tensor.shape[1] > 1:
                loss = model(tokens_tensor, labels=tokens_tensor)[0]
                ppl = np.exp(loss.cpu().detach().numpy())
                if math.isnan(ppl):
                    error_s += 1
                else:
                    ppls.append(ppl)
    return sum(ppls)/len(ppls), error_s

# Reference: https://github.com/shentianxiao/language-style-transfer/blob/f078a9d342705ea62b04dd3ac5926d3467c2ac63/code/language_model.py#L102
def calc_perplexity_token_average(model, tokenizer, predictions, device):
    """
        Calculates the token-level perplexity.
    """
    with torch.no_grad():
        nlls = []
        total = 0
        error_s = 0
        pbar = tqdm(predictions, desc='Calculating the token-level perplexity scores...')
        for i, sample in enumerate(pbar):
            text_ids = tokenizer.encode(sample, return_tensors="pt", truncation=True, max_length=512).to(device)
            if text_ids.shape[1] > 1:
                input_ids = text_ids[:, :-1]
                target_ids = text_ids[:, 1:]
                outputs = model(input_ids)
                preds = outputs.logits[0]
                calc_loss = F.nll_loss(F.log_softmax(preds, dim=1), target_ids[0])
                neg_log_likelihood = calc_loss * input_ids.shape[1]
                total += input_ids.shape[1]
                nlls.append(neg_log_likelihood)
            else:
                error_s += 1
        ppl = torch.exp(torch.stack(nlls).sum() / total)
        return ppl.item(), error_s


def all_references_in_one(ref_list):
    num_refences = len(ref_list)
    num_sentences = len(ref_list[0])
    references = []
    for i in range(num_sentences):
        arr = []
        for j in range(num_refences):
            arr.append(ref_list[j][i][0])
        references.append(arr)
    return references


def set_format(metric_name, dataset_type, exs):
    arr = None
    if metric_name == 'bleu':
        if dataset_type == 'ground_truth':
            arr = [[(ex.strip()).split(' ')] for ex in exs]
        else:
            arr = [(ex.strip()).split(' ') for ex in exs]
    else:
        if dataset_type == 'ground_truth':
            arr = [[ex.strip()] for ex in exs]
        else:
            arr = [ex.strip() for ex in exs]
    return arr

    
def read_txt_file(path):
    return open(path, 'r').readlines()


def choose_best_output_from_oracle(model, tokenizer, preds, target_style, device):
    with torch.no_grad():
        probs = []
        for sample in preds:            
            # If there is a tokenizer, then it means that we are doing sentiment classification and using BERT-classifiers
            input = tokenizer.encode(sample, return_tensors="pt", truncation=True, max_length=512).to(device)
            pred = model(input)
            positive_prob = torch.softmax(pred['logits'], dim=1)[0][1]
            target_prob = positive_prob.cpu().item()
            if target_style == 'negative':
                target_prob = 1. - target_prob
            probs.append(target_prob)
        return None, None, probs


def choose_best_output_from_mlm(model, tokenizer, preds, target_style, device):
    with torch.no_grad():
        orig_style = FROM_TO_DICT[target_style]

        orig_style_token = tokenizer.encode_plus(orig_style)['input_ids'][1]
        target_style_token = tokenizer.encode_plus(target_style)['input_ids'][1]

        probs = []
        for pred in preds:
            sentence = pred
            text = f'The following text is {tokenizer.decode(tokenizer.mask_token_id)}: {sentence}'
            input = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
            mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
            masked_output = model(**input)
            logits = masked_output.logits
            softmax = F.softmax(logits, dim = -1)
            mask_word = softmax[0, mask_index, :]
            full_probability = mask_word[0]
            orig_style_prob = full_probability[orig_style_token]
            target_style_prob = full_probability[target_style_token]
            normalized_prob = target_style_prob / (orig_style_prob  + target_style_prob)
            probs.append(normalized_prob.cpu())
        return None, None, probs


def choose_best(preds, prompt, ignore_fluency, bertscore, oracle, model, tokenizer, ppl_model, ppl_tokenizer, target_style, device):
    if len(preds) == 1:
        return preds[0]

    # Similarity score
    references = [prompt] * len(preds)

    results = bertscore.compute(predictions=preds, references=references, lang='en')
    similarity_scores = [v for v in results["f1"]]

    # Sentiment/Style-Transfer (accuracy) score
    if oracle:
        _, _, sentiment_scores = choose_best_output_from_oracle(model, tokenizer, preds, target_style, device)
    else:
        _, _, sentiment_scores = choose_best_output_from_mlm(model, tokenizer, preds, target_style, device)

    total_probs = np.array(similarity_scores) * np.array(sentiment_scores) 
    
    # Fluency score
    if not ignore_fluency:
        fluency_scores = compute_sentence_probs(ppl_model, ppl_tokenizer, preds, device)
        total_probs *= np.array(fluency_scores)
    
    max_index = np.argmax(total_probs)
    return preds[max_index]


def read_json_file(path, choose_first, ignore_fluency, bertscore, oracle, model, tokenizer, ppl_model, ppl_tokenizer, target_style, device):
    with open(path, 'r') as json_file:
        json_data = json.load(json_file)
        inputs = json_data['inputs']
        outputs =  json_data['extracted_outputs']

        if len(outputs[0]) > 1 and not choose_first:
            return [(choose_best( outputs[i], inputs[i], ignore_fluency, bertscore, oracle, model, tokenizer, ppl_model, ppl_tokenizer, target_style, device) + '\n') for i in range(len(outputs))]
        return [outputs[i][0] for i in range(len(outputs))]

def read_jsonl_file(path, choose_first, ignore_fluency, bertscore, oracle, model, tokenizer, ppl_model, ppl_tokenizer, target_style, device):
    with open(path, 'r') as json_file:
        json_list = list(json_file)

        if len(json.loads(json_list[0])['generated_output'][0]) > 1 and not choose_first:
            return [(choose_best(json.loads(line)['generated_output'], json.loads(line)['prompt'], ignore_fluency, bertscore, oracle, model, tokenizer, ppl_model, ppl_tokenizer, target_style, device) + '\n') for line in json_list]
        return [(json.loads(line)['generated_output'][0]) for line in json_list]


def read_file(path, choose_first=False, ignore_fluency=True, bertscore=None, oracle = False, model=None, tokenizer=None, ppl_model=None, ppl_tokenizer=None, target_style=None, device=None):
    if path[-5:] == 'jsonl':
        lines = read_jsonl_file(path, choose_first, ignore_fluency, bertscore, oracle, model, tokenizer, ppl_model, ppl_tokenizer, target_style, device)
    elif path[-3:] == 'txt':
        lines = read_txt_file(path)
    elif path[-4:] == 'json':
        lines = read_json_file(path, choose_first, ignore_fluency, bertscore, oracle, model, tokenizer, ppl_model, ppl_tokenizer, target_style, device)
    else:
        raise NotImplementedError('Oh no... input file is not recognized!')
    return lines


def main():
    # Parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='sacrebleu', choices=['bleu', 'sacrebleu'])
    parser.add_argument('--case_sensitive', action='store_true')
    parser.add_argument('--dataset', type=str, default='yelp_clean')
    parser.add_argument('--output_dir', type=str, default='outputs/yelp_clean/contrastive/gptj_6B/4_shot_3_samples')
    parser.add_argument('--results_save_path', type=str, default=f'results/test.tsv') #final_few_shot_yelp_clean_choosefirst_luke.tsv ail_results_shakespeare_gpt2_medium
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--choose_first', action='store_true')
    parser.add_argument('--ignore_fluency', action='store_true')
    parser.add_argument('--ppl_model', default='gpt2-large')
    parser.add_argument('--classifier', default='/nlp/scr/msuzgun/classifier/amazon_yelp/bert-base-uncased-yelp-clean-ep3')
    args = parser.parse_args()

    # Create the save directory if it does not exist
    results_save_path = args.results_save_path
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    dataset = args.dataset    
    if 'amazon' in dataset or 'yelp' in dataset:
        styles = ['positive', 'negative']
    elif 'shakespeare' in dataset:
        styles = ['Shakespearean']
    elif 'jfleg' in dataset:
        styles = ['ungrammatical']
    elif 'gyafc' in dataset:
        styles = ['informal']

    # GPU / CPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If not None, then it means that we are using the "oracle" classifier in the Prompt-and-Rerank pipeline
    oracle = args.oracle
    model_name = args.classifier

    if any(elt in dataset for elt in ['yelp', 'amazon', 'gyafc']):
        print('*** Loading the sentiment classifier...')
        # The sentiment classifiers we trained for Amazon and YELP were based on the BERT-Base model
        classifier = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
        classifier_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif 'shakespeare' in dataset:
        print('*** Loading the RoBERTa-Large style classifier...')
        roberta_classifier_dir = args.classifier
        ckpt_path = f'{roberta_classifier_dir}/checkpoint_best.pt'
        data_path = f'{roberta_classifier_dir}/shakespeare-data-bin'
        classifier = RobertaModel.from_pretrained(model_name_or_path=roberta_classifier_dir, checkpoint_file=ckpt_path, data_name_or_path=data_path).eval().to(device)
        classifier_tokenizer = None
    else:
        print('Classifier is NONE!')
        classifier = None
        classifier_tokenizer = None
        
    # If we are not using the oracle classifier, then let's use an MLM (viz., RoBERTa-Large) for calculating accuracy scores
    if oracle:
        model = classifier
        tokenizer = classifier_tokenizer
    else:
        model = AutoModelForMaskedLM.from_pretrained("roberta-large").eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")

    print(f'Loaded model. Parameters: {sum(p.numel() for p in model.parameters()):_}')

    # BERTScore
    print('*** Loading the BERTScore model...')
    bertscore = datasets.load_metric("bertscore")
    
    # Fluency model -- fluency is measured in terms of perplexity
    print(f'*** Loading the PPL model ({args.ppl_model})...')
    ppl_model = AutoModelForCausalLM.from_pretrained(args.ppl_model).eval().to(device)
    ppl_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model)

    # Reset the output file
    with open(results_save_path, 'w') as f:
        pass

    # Metric: BLEU or SacreBLEU
    if args.metric == 'bleu':
        metric = load_metric("bleu")
    else:
        metric = load_metric("sacrebleu") # preferred xmetric

    # Case sensitive (lowercase or not?)
    case_sensitive = not args.case_sensitive

    columns = []
    for style in styles:
        for tag in ['ACCURACY', 'SBLEU-REF', 'SBLEU-INPUT', 'SENT-PPL', 'TOK-PPL', 'SENT-PPL-ERROR', 'TOK-PPL-ERROR']:
            columns.append(f'{style}-{tag}')
    all_columns = ['MODEL'] + columns
    template_zero = {}
    for col in columns:
        template_zero[col] = 0.

    seen_so_far = []
    results_tsv = {}
    
    for style in styles:
        if 'amazon' in dataset or 'yelp' in dataset:
            target_style = 'positive' if style == 'negative' else 'negative'
        elif 'shakespeare' in dataset:
            target_style = 'modern' if style == 'Shakespearean' else 'Shakespearean'
        elif 'jfleg' in dataset:
            target_style = 'grammatical' if style == 'ungrammatical' else 'ungrammatical'
        elif 'gyafc' in dataset:
            target_style = 'informal' if style == 'formal' else 'formal'

        # References
        # references_path = f'datasets/{dataset}/gt_{style}_output.txt'
        # references = read_file(references_path)
        # references = set_format(args.metric, 'ground_truth', references)

        references_path_dir = f'datasets/{dataset}/gt_{style}_output*.txt'
        references_paths = glob.glob(references_path_dir,recursive=True)
        references_paths.sort()

        print(references_paths)
        references_arr = []
        for ref_path in references_paths:
            ref = read_file(ref_path)
            ref = set_format(args.metric, 'ground_truth', ref)
            references_arr.append(ref)
        references = all_references_in_one(references_arr)

        # Inputs
        input_references_path = f'datasets/{dataset}/gt_{style}_input.txt'
        input_references = read_file(input_references_path)
        input_references = set_format(args.metric, 'ground_truth', input_references)
        
        # Prediction output directory
        prediction_dir = f'{args.output_dir}/*{style}*.jsonl'
        prediction_paths = glob.glob(prediction_dir,recursive=True)
        prediction_paths.sort()

        # Run 
        for path in prediction_paths:
            print(path)
            model_name, delimeter_type = (path.split('/')[-1]).split(f'_{style}_')
            delimeter_type = delimeter_type.split('.jsonl')[0]
            predictions = read_file(path, args.choose_first, args.ignore_fluency, bertscore, oracle, model, tokenizer, ppl_model, ppl_tokenizer, target_style, device)
            predictions = set_format(args.metric, 'prediction', predictions)

            tmp = model_name + '_' + delimeter_type

            if not(tmp in seen_so_far):
                seen_so_far.append(tmp)
                results_tsv[tmp] = {}
                for col in columns:
                    results_tsv[tmp][col] = 0.

            try:
                # Calculate the sacreBLEU scores w.r.t. predictions
                result = calc_blue(metric, predictions, references, case_sensitive)
                results_tsv[tmp][style + '-SBLEU-REF'] = result['score']

                # Calculate the sacreBLEU scores w.r.t. inputs
                result = calc_blue(metric, predictions, input_references, case_sensitive)
                results_tsv[tmp][style + '-SBLEU-INPUT'] = result['score']

                # Calculate the style-transfer (sentiment) accuracy score of predictions
                _, acc, _ = calc_sentiment_accuracy(classifier, classifier_tokenizer, predictions, target_style, device)
                results_tsv[tmp][style + '-ACCURACY'] = acc

                # Calculuate the sentence-level and token-level perplexity scores of predictions
                avg_ppl_tok, err_tok_level = calc_perplexity_token_average(ppl_model, ppl_tokenizer, predictions, device)
                results_tsv[tmp][style + '-TOK-PPL'] = avg_ppl_tok
                results_tsv[tmp][style + '-TOK-PPL-ERROR'] = err_tok_level
                avg_ppl_sent, err_sent_level = calc_perplexity_sent_average(ppl_model, ppl_tokenizer, predictions, device)
                results_tsv[tmp][style + '-SENT-PPL'] = avg_ppl_sent
                results_tsv[tmp][style + '-SENT-PPL-ERROR'] = err_sent_level

                print(f'{tmp}... acc: {acc} ; sent-ppl: {avg_ppl_sent}; token-ppl: {avg_ppl_tok}')

            except:
                print('****')
                print(path)
                print('****')

    with open(results_save_path, 'w') as output_file:
        tsv_writer = csv.DictWriter(output_file, fieldnames=all_columns)
        tsv_writer.writeheader()
        for model in seen_so_far:
            tmp_dict = results_tsv[model]
            tmp_dict['MODEL'] = model
            tsv_writer.writerow(tmp_dict)
    
if __name__ == "__main__":
    main()
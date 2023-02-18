import pandas as pd
import transformers
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.cuda.amp as amp
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import os
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm
import random
import heapq
from collections import defaultdict
# from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

SPAM_COLLECTION = './SMSSpamCollection.txt'
TARGET_NAMES = ['Ham', 'Spam']


def split_collection():
    print("Processing the spam collection data...")
    spams = []
    hams = []
    with open(SPAM_COLLECTION, encoding="utf8") as f:
        for line in f.readlines():
            words = line.split("	")
            if words[0] == "ham":
                hams.append(words[1].strip())
            elif words[0] == "spam":
                spams.append(words[1].strip())
    return hams, spams


def roberta_classifier(spam, ham, batch_size=16, num_epochs=50):
    # Set the device and set the seed value for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Load the RoBerta model
    if os.path.isfile('./roberta_clean'):
        model = transformers.RobertaForSequenceClassification.from_pretrained('./roberta_clean').cuda()
    else:
        model = transformers.RobertaForSequenceClassification.from_pretrained('roberta-base').cuda()
        model.save_pretrained('roberta_clean')

    for param in model.roberta.parameters():
        param.requires_grad = False
    # Tokenize the sentences using the RoBERTa tokenizer
    tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')

    max_length = max(len(max(ham, key=len)), len(max(spam, key=len)))
    if max_length > 400:
        max_length = 400

    ham_inputs = tokenizer.batch_encode_plus(ham, max_length=max_length, truncation=True, pad_to_max_length=True,
                                             return_tensors='pt')
    spam_inputs = tokenizer.batch_encode_plus(spam, max_length=max_length, truncation=True, pad_to_max_length=True,
                                              return_tensors='pt')

    # Define the labels for each domain (0 for ham, 1 forspam)
    ham_labels = torch.tensor([0] * len(ham)).cuda()
    spam_labels = torch.tensor([1] * len(spam)).cuda()

    # Concatenate the inputs and labels for both domains
    inputs = torch.cat([ham_inputs['input_ids'], spam_inputs['input_ids']], dim=0).cuda()
    attention_mask = torch.cat([ham_inputs['attention_mask'], spam_inputs['attention_mask']], dim=0).cuda()
    labels = torch.cat([ham_labels, spam_labels], dim=0)

    # Define a TensorDataset with the inputs and labels
    dataset = TensorDataset(inputs, attention_mask, labels)

    # Create a DataLoader from the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up the optimizer and criterion
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: min(x / 20, 1) * 1e-6)

    # scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))
    # criterion = nn.CrossEntropyLoss()

    # Fine-tune the model
    model.train()
    loss_lst = []
    print("Start training RoBerta spam classifier...")
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        epoch_predictions = []
        epoch_labels = []
        for batch in dataloader:
            batch_inputs_ids, batch_attention_mask, batch_labels = batch
            with amp.autocast():
                output = model(input_ids=batch_inputs_ids, attention_mask=batch_attention_mask.to(device),
                               labels=batch_labels.to(device))
                batch_logits = output['logits']
                batch_predictions = batch_logits.argmax(dim=1)
                batch_loss = output[0]
            total_loss += batch_loss.item()
            with amp.autocast():
                batch_loss.backward()

            optimizer.step()
            # if epoch < 20:
            # scheduler1.step()
            # else:
            # scheduler2.step()

            optimizer.zero_grad()
            # Tracking the predictions and labels
            epoch_predictions += batch_predictions.to("cpu")
            epoch_labels += batch_labels.to("cpu")

        print(classification_report(epoch_labels, epoch_predictions, target_names=TARGET_NAMES))
        loss_lst.append(total_loss / len(dataloader))
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    print("Finished training RoBerta spam classifier...")
    return model


def polynomial_decay(iteration):
    return (1 - iteration / 30000) ** 0.9


def bart_ham(ham):
    # Set the device and set the seed value for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Load the BART model
    if os.path.isfile('./bart_clean'):
        model = transformers.BartForConditionalGeneration.from_pretrained("./bart_clean").cuda()
    else:
        model = transformers.BartForConditionalGeneration.from_pretrained("facebook/bart-base",
                                                                          forced_bos_token_id=0).cuda()
        model.save_pretrained('bart_new')

    # Freeze the model layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 3 layers in the encoder
    for param in model.model.encoder.layers[-3:].parameters():
        param.requires_grad = True
    # Unfreeze the last 3 layers in the decoder
    for param in model.model.decoder.layers[-3:].parameters():
        param.requires_grad = True

    tokenizer = transformers.BartTokenizer.from_pretrained("facebook/bart-base")

    # Create a dataset for the domain	
    tokenized_data = tokenizer.batch_encode_plus(ham, pad_to_max_length=True,
                                                 return_tensors='pt',
                                                 truncation=True)

    inputs = tokenized_data['input_ids'].cuda()
    attention_mask = tokenized_data['attention_mask'].cuda()

    dataset = TensorDataset(inputs, attention_mask)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Set up the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    # Define the linear warmup schedule
    # scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x / 20, 1) * 3e-5)

    # Define the polynomial decay schedule

    # scheduler2 = optim.lr_scheduler.LambdaLR(optimizer, polynomial_decay)

    # Set the dropout
    dropout = 0.3

    # Fine-tune the model
    model.train()
    print("Start fine tuning Bart model...")
    for epoch in tqdm(range(2)):
        print(f"\n Currently at epoch {epoch}...")
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with amp.autocast():
                # output = model(input_ids=input_ids, attention_mask=attention_mask, dropout=dropout)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output[0]
                loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            if loss.device != device:
                loss = loss.to(device)
            total_loss += loss.item()
            with amp.autocast():
                loss.backward()
            optimizer.step()
            # if epoch < 20:
            #     scheduler1.step()
            # else:
            #     scheduler2.step()
            optimizer.zero_grad()
        print(f'Iteration {epoch + 1}, Loss: {total_loss / len(dataloader)}')
        # Creates a checkpoint after each 10 epochs
        if epoch % 10 == 0:
            model.save_pretrained('bart_trained')
    print("Finished training ham Bart...")
    return model


def corruption(word):
    abc = "abcdefghijklmnopqrstuvwxyz"
    p = random.random()
    index = random.choice(range(len(word)))
    if p < 1 / 3:
        word = word[:index] + word[index:]
    elif p < 2 / 3:
        word = word[:index] + random.choice(abc) + word[index:]
    else:
        word = word[:index] + word[index + 1:]
    return word


def add_mask(classifier, lm, sentences, k, choosing_baseline=False, replacing_baseline=False,
             first_attention_layer=False):
    # roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    roberta_tokenizer = transformers.RobertaTokenizerFast.from_pretrained("roberta-base")
    bart_tokenizer = transformers.BartTokenizer.from_pretrained("facebook/bart-large")
    fake_sentences = []
    for sentence in sentences:
        if choosing_baseline:  # choose k words according to the score after deleting the word from the sentence
            scores = {}
            list_sentence = sentence.split(" ")
            for i in range(len(list_sentence)):
                if i == len(list_sentence) - 1:
                    new_sentence = " ".join(list_sentence[:-1])
                else:
                    new_sentence = " ".join(list_sentence[:i] + list_sentence[i + 1:])
                input_ids = torch.tensor(roberta_tokenizer.encode(new_sentence)).unsqueeze(0)
                output = classifier(input_ids)
                scores[i] = output["logits"][0][1].item()
            indices = heapq.nsmallest(k, scores, key=scores.get)
            for index in indices:
                list_sentence[index] = bart_tokenizer.mask_token
            masked_sentence = " ".join(list_sentence)
            input_ids = torch.tensor(bart_tokenizer.encode(masked_sentence)).unsqueeze(0)
            # Generate the completions for the masked tokens
            new_sentece_ids = lm.generate(input_ids, max_length=len(input_ids[0]) * 2, num_beams=5)
            fake_sentences.append(bart_tokenizer.decode(new_sentece_ids[0], skip_special_tokens=True))

        elif replacing_baseline:  # choose k words according to the attention score and corrupt them

            # Tokenize the sentence
            # input_ids = roberta_tokenizer.encode(sentence, return_tensors='pt')
            input_ids = roberta_tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True)
            # Run the sentence in the model inference
            outputs = classifier(input_ids["input_ids"], output_attentions=True)
            # Extract the last attention layer
            last_hidden_states = outputs.attentions[-1][0][0][0]

            # Get the k words with the highest attention

            sentence_lst = sentence.split(" ")

            ##########################################################
            sentence_dict = {}
            counter = 0
            for i, word in enumerate(sentence.split(" ")):
                sentence_dict[(counter, counter + len(word))] = i
                counter = counter + len(word) + 1
            offsets = input_ids["offset_mapping"].squeeze(0)
            offsets_dicts = {(item[0].item(), item[1].item()): i for i, item in enumerate(offsets)}

            def map_words(dict1, dict2):
                new_dict = {}
                for key1, value1 in dict1.items():
                    words = []
                    for key2, value2 in dict2.items():
                        if key1[0] <= key2[0] and key2[1] <= key1[1]:
                            if key2 == (0, 0):
                                if -1 in new_dict:
                                    new_dict[-1].append(value2)
                                else:
                                    new_dict[-1] = [value2]
                                continue
                            words.append(value2)
                        if key2[0] > key1[1]:
                            break
                    new_dict[value1] = words
                return new_dict

            holy_dict = map_words(sentence_dict, offsets_dicts)
            holy_dict[-1].append(0)
            reversed_dict = {v: k for k, values in holy_dict.items() for v in values}

            # Get the k words with the highest attention
            attention_order = np.argsort(last_hidden_states.detach(), axis=-1).tolist()
            mask_indices = set()
            for i in range(len(attention_order) - 1, -1, -1):
                if reversed_dict[attention_order[i]] == -1:
                    continue
                if len(mask_indices) == k:
                    break
                mask_indices.add(reversed_dict[attention_order[i]])
                counter += 1

            ##########################################################

            for index in mask_indices:
                sentence_lst[index] = corruption(sentence_lst[index])

            fake_sentences.append(" ".join(sentence_lst))
        else:
            # Tokenize the sentence
            # input_ids = roberta_tokenizer.encode(sentence, return_tensors='pt')
            input_ids = roberta_tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True)

            # Run the sentence in the model inference
            outputs = classifier(input_ids["input_ids"], output_attentions=True)

            if first_attention_layer:
                # Extract the last attention layer
                last_hidden_states = outputs.attentions[0][0][0][0]
            else:
                # Extract the last attention layer
                last_hidden_states = outputs.attentions[-1][0][0][0]

            # Get the k words with the highest attention
            sentence_lst = sentence.split(" ")

            ##########################################################
            sentence_dict = {}  # maps indices to the relevant word
            counter = 0
            for i, word in enumerate(sentence.split(" ")):
                sentence_dict[(counter, counter + len(word))] = i
                counter = counter + len(word) + 1
            offsets = input_ids["offset_mapping"].squeeze(0)

            offsets_dicts, i = {}, 0
            for item in offsets:
                indices_obj = (item[0].item(), item[1].item())
                if indices_obj not in offsets_dicts:
                    offsets_dicts[indices_obj] = i
                    i += 1

            # offsets_dicts = {(item[0].item(), item[1].item()): i for i, item in enumerate(offsets)} # bug with offsets object

            def map_words(dict1, dict2):
                new_dict = {}
                for key1, value1 in dict1.items():
                    words = []
                    for key2, value2 in dict2.items():
                        if key1[0] <= key2[0] and key2[1] <= key1[1]:
                            if key2 == (0, 0):
                                if -1 in new_dict:
                                    new_dict[-1].append(value2)
                                else:
                                    new_dict[-1] = [value2]
                                continue
                            words.append(value2)
                        if key2[0] > key1[1]:
                            break
                    new_dict[value1] = words
                return new_dict

            holy_dict = map_words(sentence_dict,
                                  offsets_dicts)  # a dict of a map from the original word in the sentence to whole of its tokens after tokenization
            holy_dict[-1].append(0)
            reversed_dict = {v: k for k, values in holy_dict.items() for v in values}

            # Get the k words with the highest attention
            attention_order = np.argsort(last_hidden_states.detach(), axis=-1).tolist()
            mask_indices = set()
            for i in range(len(attention_order) - 1, -1, -1):
                try:
                    if attention_order[i] >= max(reversed_dict):
                        continue
                    if reversed_dict[attention_order[i]] == -1:
                        continue
                    if len(mask_indices) == k:
                        break
                    mask_indices.add(reversed_dict[attention_order[i]])
                    counter += 1
                except:
                    print(f"The sentence {sentence} could not be process")

            ##########################################################

            for index in mask_indices:
                sentence_lst[index] = bart_tokenizer.mask_token

            masked_sentence = " ".join(sentence_lst)
            input_ids = torch.tensor(bart_tokenizer.encode(masked_sentence)).unsqueeze(0)

            # Generate the completions for the masked tokens
            new_sentence_ids = lm.generate(input_ids, max_length=len(input_ids[0]) * 2, num_beams=5)
            fake_sentences.append(bart_tokenizer.decode(new_sentence_ids[0], skip_special_tokens=True))

    return fake_sentences


def model_performance(sentences, fake_sentences, k, cur_method='regular'):
    sentences_df = pd.DataFrame({"original_sentences": sentences, "adversarial_sentences": fake_sentences})
    sentences_df.to_csv(f"{cur_method}_sentences_{k}.csv")

    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    model = transformers.RobertaForSequenceClassification.from_pretrained(
        'mariagrandury/roberta-base-finetuned-sms-spam-detection')

    input_ids = torch.tensor(
        [tokenizer.encode(sentence, add_special_tokens=True, pad_to_max_length=True, max_length=200, truncation=True)
         for sentence in sentences])
    attention_mask = torch.where(input_ids == 1, torch.tensor(0), torch.tensor(1))
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)[0]
        prediction = torch.softmax(logits, dim=1)[:, 1]

    mean_scores = prediction.mean()
    print("Original scores mean: {:.2f}%".format(mean_scores * 100))

    input_ids = torch.tensor(
        [tokenizer.encode(sentence, add_special_tokens=True, pad_to_max_length=True, max_length=200, truncation=True)
         for sentence in fake_sentences])
    attention_mask = torch.where(input_ids == 1, torch.tensor(0), torch.tensor(1))
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)["logits"]
        prediction = torch.softmax(logits, dim=1)[:, 1]

    mean_scores = prediction.mean()
    print("Fake scores mean: {:.2f}%".format(mean_scores * 100))


def similar_performance(sentences, fake_sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    similarity = 0
    counter = 0
    for original, fake in zip(sentences, fake_sentences):
        # Compute embedding for both lists
        original_embedding = model.encode(original, convert_to_tensor=True)
        fake_embedding = model.encode(fake, convert_to_tensor=True)
        similarity += util.pytorch_cos_sim(original_embedding, fake_embedding).item()
        counter += 1
    print(f"Similarity measure: {similarity / counter}")


def evaluate_pred(pred, refer):
    bleu = load("bleu")
    rouge = load('rouge')
    google_bleu = load("google_bleu")
    bertscore = load("bertscore")
    score = dict()
    score["bleu"] = bleu.compute(predictions=pred, references=refer)
    score["rouge"] = rouge.compute(predictions=pred, references=refer)
    score["google_bleu"] = google_bleu.compute(predictions=pred, references=refer)
    # score["bertscore"] = bertscore.compute(predictions=pred, references=refer,lang="en")
    return score


if __name__ == '__main__':
    hams, spams = split_collection()

    # Train RoBerta
    # roberta_model = roberta_classifier(spams, hams)
    # roberta_model.save_pretrained('roberta_trained')

    # Train Bart
    # bart_model = bart_ham(hams)
    # bart_model.save_pretrained('bart_trained')

    # Create adversarial attacks

    # Load trained Roberta model
    roberta_model = transformers.RobertaForSequenceClassification.from_pretrained('./roberta_trained')
    # # roberta_model = transformers.RobertaModel.from_pretrained('./roberta_clean')

    # Load trained Bart model
    # bart_model = transformers.BartForConditionalGeneration.from_pretrained("./bart_trained")
    bart_model = transformers.BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0)

    ####################################################################################################################
    # Choose hyper-parameters
    choose_k = False
    first_attention_layer = False

    if choose_k:
        for k in range(7, 8):
            fake_spams = add_mask(roberta_model, bart_model, spams, k, first_attention_layer)
            print(f"Treshold: {k}")
            model_performance(spams, fake_spams, k)
            similar_performance(spams, fake_spams)
            print(evaluate_pred(spams, fake_spams))

    ####################################################################################################################
    test_exp = True
    methods = ['our', 'choosing_baseline']  # ['replacing_baseline']
    replacing_baseline, choosing_baseline = False, False
    if test_exp:
        for method in methods:
            if method == 'choosing_baseline':
                choosing_baseline = True
            elif method == 'replacing_baseline':
                replacing_baseline = True
            t = time.time()
            fake_spams = add_mask(roberta_model, bart_model, spams[:100], k=7, choosing_baseline=choosing_baseline,
                                  replacing_baseline=replacing_baseline)
            print(f"{method} time:" , time.time() - t)
            # print(f"Treshold: {7}")
            # model_performance(spams, fake_spams, 7, cur_method=method)
            # similar_performance(spams, fake_spams)
            # print(evaluate_pred(spams, fake_spams))

            # Shut down the baseline activation
            replacing_baseline, choosing_baseline = False, False

    ####################################################################################################################
    ranodm_test = False
    if ranodm_test:
        spams = ['100% original medical boxes only NIS 500\n for orders \n https://wa.me/972504901774']
        k = 3
        # for method in methods:
        #     if method == 'choosing_baseline':
        #         choosing_baseline = True
        #     elif method == 'replacing_baseline':
        #         replacing_baseline = True
        method = 'random_test'
        fake_spams = add_mask(roberta_model, bart_model, spams[:10], k=k, choosing_baseline=choosing_baseline,
                              replacing_baseline=replacing_baseline)
        print(f"Treshold: {k}")
        model_performance(spams[:10], fake_spams, k, cur_method=method)
        similar_performance(spams[:10], fake_spams)
        print(evaluate_pred(spams[:10], fake_spams))

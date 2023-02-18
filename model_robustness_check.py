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
from evaluate import load
import math

TARGET_NAMES = ['Ham', 'Spam']
SPAM_COLLECTION = './SMSSpamCollection.txt'


def train_classifier(train_data, hams_df, batch_size=16, num_epochs=30, force_retraining=False, model_type='bert'):
    if model_type == 'roberta':
        num_epochs = 1

        # Set the device and set the seed value for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    # Load the Bert model
    if not force_retraining:
        if model_type == 'bert':
            if os.path.isdir('./bert_spam_detection_retrained'):
                model = transformers.BertForSequenceClassification.from_pretrained(
                    './bert_spam_detection_retrained').cuda()
                print("Bert retrained was loaded...")
                return model
        else:
            if os.path.isdir('./roberta_spam_detection_retrained'):
                model = transformers.RobertaForSequenceClassification.from_pretrained(
                    './roberta_spam_detection_retrained').cuda()
                print("Roberta retrained was loaded...")
                return model

    if model_type == 'bert':
        if os.path.isdir('./bert_spam_detection_clean'):
            model = transformers.BertForSequenceClassification.from_pretrained('./bert_spam_detection_clean').cuda()
        else:
            model = transformers.BertForSequenceClassification.from_pretrained(
                'mrm8488/bert-tiny-finetuned-enron-spam-detection').cuda()
            model.save_pretrained('bert_spam_detection_clean')
    else:
        if os.path.isdir('./roberta_trained'):
            model = transformers.RobertaForSequenceClassification.from_pretrained('./roberta_trained').cuda()
        else:
            print("Trained RoBerta does not exists")
            return

    # Freezing Bert model weights, and training only the spam classifier
    if model_type == 'bert':
        for param in model.bert.parameters():
            param.requires_grad = False
    else:
        for param in model.roberta.parameters():
            param.requires_grad = False

    if model_type == 'bert':
        # Tokenize the sentences using the RoBERTa tokenizer
        tokenizer = transformers.BertTokenizer.from_pretrained('mrm8488/bert-tiny-finetuned-enron-spam-detection')
    else:
        # Tokenize the sentences using the RoBERTa tokenizer
        tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')

    max_length = max(len(max(hams_df.iloc[:, 0], key=len)), len(max(train_data, key=len)))
    if max_length > 400:
        max_length = 400

    train_inputs = tokenizer.batch_encode_plus(train_data, max_length=max_length, truncation=True,
                                               pad_to_max_length=True,
                                               return_tensors='pt')

    hams_inputs = tokenizer.batch_encode_plus(hams_df.iloc[:, 0], max_length=max_length, truncation=True,
                                              pad_to_max_length=True,
                                              return_tensors='pt')

    # Define the labels for each domain (0 for ham, 1 for spam) - both datasets are spam only
    train_labels = torch.tensor([1] * len(train_data))
    hams_labels = torch.tensor([0] * hams_df.shape[0])

    inputs = torch.cat([hams_inputs['input_ids'], train_inputs['input_ids']], dim=0).cuda()
    attention_mask = torch.cat([hams_inputs['attention_mask'], train_inputs['attention_mask']], dim=0).cuda()
    labels = torch.cat([hams_labels, train_labels], dim=0)

    # Define a TensorDataset with the inputs and labels
    # dataset = TensorDataset(train_inputs['input_ids'].cuda(), train_inputs['attention_mask'].cuda(), train_labels)
    dataset = TensorDataset(inputs, attention_mask, labels)

    # Create a DataLoader from the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up the optimizer and criterion
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    # Fine-tune the model
    model.train()
    loss_lst = []
    print(f"Start training {model_type} spam classifier...")
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

            optimizer.zero_grad()
            # Tracking the predictions and labels
            epoch_predictions += batch_predictions.to("cpu")
            epoch_labels += batch_labels.to("cpu")

        print(classification_report(epoch_labels, epoch_predictions))
        loss_lst.append(total_loss / len(dataloader))
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    print(f"Finished training {model_type} spam classifier...")
    model.save_pretrained(f'{model_type}_spam_detection_retrained')
    return model


def model_performances(model, test_dataset, model_type, type='spam', language_model='bert'):
    print(f"Evaluating on the {model_type} {language_model} model for spam detection")
    if language_model == 'bert':
        tokenizer = transformers.BertTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-enron-spam-detection")
    else:
        tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')

    max_length = max(1, len(max(test_dataset.iloc[:, 0], key=len)))
    if max_length > 400:
        max_length = 400

    input_ids = torch.tensor(
        [tokenizer.encode(sentence, add_special_tokens=True, pad_to_max_length=True, max_length=max_length,
                          truncation=True)
         for sentence in test_dataset.iloc[:, 0]])
    if language_model == 'bert':
        attention_mask = torch.where(input_ids == 0, torch.tensor(0), torch.tensor(1))
    else:
        attention_mask = torch.where(input_ids == 1, torch.tensor(0), torch.tensor(1))

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    prediction = []
    with torch.no_grad():
        for batch in dataloader:
            batch_inputs_ids, batch_attention_mask = batch
            logits = model(batch_inputs_ids.cuda(), attention_mask=batch_attention_mask.cuda())["logits"]
            if type == 'spam':
                prediction += torch.softmax(logits, dim=1)[:, 1].tolist()
            else:
                prediction += torch.softmax(logits, dim=1)[:, 0].tolist()

    mean_scores = sum(prediction) / len(prediction)
    print("Adversarial sentences mean scores: {:.2f}%".format(mean_scores * 100))


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


if __name__ == '__main__':
    # End of the circle - we attacked a spam detection model, and now we want to retrain it
    data_path = './data/our_sentences_7.csv'
    model_type = 'roberta'
    data = pd.read_csv(data_path)
    adversarial_data = data.iloc[:, 2]  # extracting the adversarial sentences
    p = 0.7  # data percentage to retraining
    adversarial_train_dataset = adversarial_data.iloc[:math.floor((adversarial_data.shape[0] * p))]
    adversarial_test_dataset = adversarial_data.iloc[math.floor((adversarial_data.shape[0] * p)) + 1:]
    adversarial_test_dataset = pd.DataFrame({'spam sentences': adversarial_test_dataset})
    # Load the original spams collection
    hams, spams = split_collection()
    spams_df = pd.DataFrame({'spam sentences': spams})
    hams_df = pd.DataFrame({'ham sentences': hams[:1]})
    test_ham = pd.DataFrame({'ham sentences': hams[1:]})

    test_dataset = spams_df.iloc[math.floor((spams_df.shape[0] * p)) + 1:]

    # Load clean Bert for spam detection
    if model_type == 'roberta':
        if os.path.isdir('./roberta_trained'):
            clean_model = transformers.RobertaForSequenceClassification.from_pretrained('./roberta_trained').cuda()
        else:
            print("Trained RoBerta does not exists")
    else:
        if os.path.isdir('./bert_spam_detection_clean'):
            clean_model = transformers.BertForSequenceClassification.from_pretrained(
                './bert_spam_detection_clean').cuda()
        else:
            clean_model = transformers.BertForSequenceClassification.from_pretrained(
                'mrm8488/bert-tiny-finetuned-enron-spam-detection').cuda()
            clean_model.save_pretrained('bert_spam_detection_clean')

    # Train Bert on p percentages of the adversarial spam messages
    retrained_model = train_classifier(adversarial_train_dataset, hams_df, model_type=model_type, force_retraining=True)
    models = [clean_model, retrained_model]
    # Evaluate performance on the 1-p percentages of the adversarial messages
    print("\n Testing the model on the original spams messages:")
    for model in models:
        model_performances(model, test_dataset, 'clean' if model == clean_model else 'retrained',
                           language_model=model_type)

    # Evaluate performance on the 1-p percentages of the adversarial messages
    print("\n Testing the model on the adversarial spams messages:")
    for model in models:
        model_performances(model, adversarial_test_dataset, 'clean' if model == clean_model else 'retrained',
                           language_model=model_type)
    ################################################################################################################

    print("\n Testing the model on the original hams messages:")
    for model in models:
        model_performances(model, test_ham, 'clean' if model == clean_model else 'retrained', type='ham',
                           language_model=model_type)

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
                hams.append(words[1])
            elif words[0] == "spam":
                spams.append(words[1])
    return hams, spams


def roberta_classifier(spam, ham, batch_size=16, num_epochs=60):
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
    optimizer = optim.Adam(model.classifier.parameters(), lr=0)

    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: min(x / 20, 1) * 1e-6)

    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

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
            if epoch < 20:
                scheduler1.step()
            else:
                scheduler2.step()

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
        model = transformers.BartModel.from_pretrained('./bart_clean').cuda()
    else:
        model = transformers.BartModel.from_pretrained("facebook/bart-base").cuda()
        model.save_pretrained('bart_clean')

    tokenizer = transformers.RobertaTokenizer.from_pretrained('facebook/bart-base')

    # Create a dataset for the domain
    tokenized_data = tokenizer.batch_encode_plus(ham, pad_to_max_length=True,
                                          return_tensors='pt',
                                          truncation=True)

    inputs = tokenized_data['input_ids'].cuda()
    attention_mask = tokenized_data['attention_mask'].cuda()

    dataset = TensorDataset(inputs, attention_mask)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)

    # Set up the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    # Define the linear warmup schedule
    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x / 20, 1) * 3e-5)

    # Define the polynomial decay schedule

    scheduler2 = optim.lr_scheduler.LambdaLR(optimizer, polynomial_decay)

    # Set the dropout
    dropout = 0.3

    # Fine-tune the model
    model.train()
    print("Start fine tuning Bart model...")
    for epoch in tqdm(range(30)):
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
            if epoch < 20:
                scheduler1.step()
            else:
                scheduler2.step()
            optimizer.zero_grad()
        print(f'Iteration {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    print("Finished training RoBerta spam classifier...")
    return model


def add_mask(classifier, lm, sentences, k):
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    fake_sentences = []
    for sentence in sentences:
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
        outputs = classifier(input_ids)
        last_hidden_states = outputs[0]

        # Get the k words with the highest attention
        indices = np.argsort(last_hidden_states[0], axis=-1)[-k:]
        for index in indices:
            input_ids[0][index] = tokenizer.mask_token_id

        # Generate the completions for the masked tokens
        new_sentece_ids = lm.generate(input_ids)
        fake_sentences.append(tokenizer.decode(new_sentece_ids[0], skip_special_tokens=True))
    return fake_sentences


def add_msak_baseline(classifier, lm, sentences, k):
    pass


def model_performance(sentences, fake_sentences):
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    model = transformers.RobertaForSequenceClassification.from_pretrained(
        'mariagrandury/roberta-base-finetuned-sms-spam-detection').cuda()
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences])

    with torch.no_grad():
        logits = model(input_ids)[0]
        predictions = torch.argmax(logits, dim=1)

    correct_predictions = (predictions == 1).sum().item()

    # Print the accuracy
    accuracy = correct_predictions / len(sentences)
    print("Original Accuracy: {:.2f}%".format(accuracy * 100))

    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True) for sentence in fake_sentences])

    with torch.no_grad():
        logits = model(input_ids)[0]
        predictions = torch.argmax(logits, dim=1)

    correct_predictions = (predictions == 1).sum().item()

    # Print the accuracy
    accuracy = correct_predictions / len(sentences)
    print("Fake Accuracy: {:.2f}%".format(accuracy * 100))


def similar_performance(sentences, fake_sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    similarity = 0
    counter = 0
    for original, fake in zip(sentences, fake_sentences):
        # Compute embedding for both lists
        original_embedding = model.encode(original, convert_to_tensor=True)
        fake_embedding = model.encode(fake, convert_to_tensor=True)
        similarity += util.pytorch_cos_sim(original_embedding, fake_embedding)
        counter += 1
    print(f"Similarity measure: {similarity / counter}")


if __name__ == '__main__':
    hams, spams = split_collection()
    # roberta_model = roberta_classifier(spams, hams)
    # roberta_model.save_pretrained('roberta_trained')
    bart_model = bart_ham(hams)
    bart_model.save_pretrained('bart_trained')

    # for k in range(6):
    #     fake_spams = add_mask(roberta_model, bart_model, spams, k)
    #     print("Treshold: {k}")
    #     model_performance(spams, fake_spams)
    #     similar_performance(spams, fake_spams)

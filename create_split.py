import transformers
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.cuda.amp as amp
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def split_collection():
    spams = []
    hams = []
    with open(r"C:\Users\noysc\Desktop\SMSSpamCollection.txt", encoding="utf8") as f:
        for line in f.readlines():
            words = line.split("	")
            if words[0] == "ham":
                hams.append(words[1])
            elif words[0] == "spam":
                spams.append(words[1])
    return hams, spams


# with open(r"C:\Users\noysc\Desktop\spam.txt", "w", encoding="utf8") as f:
# 	f.writelines(spams)
# with open(r"C:\Users\noysc\Desktop\hams.txt", "w", encoding="utf8") as f:
# 	f.writelines(hams)


def roberta_classifier(spam, ham, batch_size=256, num_epochs=5000):
    # Set the device and set the seed value for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Load the RoBerta model
    model = transformers.RobertaModel.from_pretrained('roberta-base').cuda()

    # Tokenize the sentences using the RoBERTa tokenizer
    tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')

    ham_inputs = tokenizer(ham, return_tensors='pt', padding=True, truncation=True)
    spam_inputs = tokenizer(spam, return_tensors='pt', padding=True, truncation=True)

    # Define the labels for each domain (0 for ham, 1 forspam)
    ham_labels = torch.tensor([0] * len(ham)).cuda()
    spam_labels = torch.tensor([1] * len(spam)).cuda()

    # Concatenate the inputs and labels for both domains
    inputs = torch.cat([ham_inputs['input_ids'], spam_inputs['input_ids']], dim=0).cuda()
    labels = torch.cat([ham_labels, spam_labels], dim=0)

    # Define a TensorDataset with the inputs and labels
    dataset = TensorDataset(inputs, labels)

    # Create a DataLoader from the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0).cuda()
    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: min(x / 100, 1) * 1e-6)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))
    criterion = nn.CrossEntropyLoss()

    # Fine-tune the model
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with amp.autocast():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(output[0], input_ids.argmax(dim=2))
            total_loss += loss.item()
            with amp.autocast():
                loss.backward()
            optimizer.step()
            if epoch < 100:
                scheduler1.step()
            else:
                scheduler2.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')
    return model


def bart_ham(ham):
    # Set the device and set the seed value for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Load the BART model
    model = transformers.BartModel.from_pretrained('bart-base')

    # Create a dataset for the domain
    dataset = transformers.BartTokenizer.from_pretrained('bart-base').batch_encode_plus(ham, pad_to_max_length=True,
                                                                                        return_tensors='pt')

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3000, shuffle=True)

    # Set up the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    # Define the linear warmup schedule
    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x / 5000, 1) * 3e-5)

    # Define the polynomial decay schedule
    def polynomial_decay(iteration):
        return (1 - iteration / 30000) ** 0.9

    scheduler2 = optim.lr_scheduler.LambdaLR(optimizer, polynomial_decay)

    # Set the dropout
    dropout = 0.3

    # Fine-tune the model
    model.train()
    for iteration in range(30000):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with amp.autocast():
                output = model(input_ids=input_ids, attention_mask=attention_mask, dropout=dropout)
                loss = criterion(output[0], input_ids.argmax(dim=2))
            total_loss += loss.item()
            with amp.autocast():
                loss.backward()
            optimizer.step()
            if iteration < 5000:
                scheduler1.step()
            else:
                scheduler2.step()
            optimizer.zero_grad()
        print(f'Iteration {iteration + 1}, Loss: {total_loss / len(dataloader)}')

    return model


def add_mask(classifier, lm, sentence, k):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    outputs = classifier(input_ids)
    last_hidden_states = outputs[0]

    # Get the k words with the highest attention
    indices = np.argsort(last_hidden_states[0], axis=-1)[-k:]
    for index in indices:
        input_ids[0][index] = tokenizer.mask_token_id

    # Generate the completions for the masked tokens
    new_sentece_ids = lm.generate(input_ids)
    completions_text = tokenizer.decode(new_sentece_ids[0], skip_special_tokens=True)
    return completions_text


def performance(sentences, fake_sentences):
    pass


if __name__ == '__main__':
    hams, spams = split_collection()
    roberta_model = roberta_classifier(spams, hams)
    bart_model = bart_ham(hams)
    k = 1
    for sentence in spams:
        new_sentence = add_mask(roberta_model, bart_model, sentence, k)

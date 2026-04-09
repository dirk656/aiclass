import csv

import torch 
from torch.utils.data import DataLoader , TensorDataset
from collections import Counter

def load_file(path):

    texts , labels = [] , []
    with open(path , 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
        
        return texts , labels
    
def build_voc(texts):
    counter = Counter(word for text in texts for word in text.split())

    vocab = {"<PAD>":0 , "<UNK>":1}

    for word in counter:
        vocab[word] = len(vocab)
    
    return vocab 


def text_to_ids(texts ,vocab , max_len):
    words = texts.split()

    ids = [vocab.get(w,1) for w in words]

    if len(ids) > max_len:
        return ids[:max_len]
    
    else :
        return ids + [0] * (max_len - len(ids))
    

def main () -> None:
    train_texts , train_labels = load_file("csvtextdata/train.csv")
    test_texts , test_labels = load_file("csvtextdata/test.csv")

    vocab = build_voc(train_texts)

    max_len = 20
    batch_size = 2 

    x_train = torch.tensor([text_to_ids(texts=t, vocab=vocab, max_len=max_len) for t in train_texts], dtype=torch.long)
    y_train = torch.tensor(train_labels, dtype=torch.long)

    x_test = torch.tensor([text_to_ids(texts=t, vocab=vocab, max_len=max_len) for t in test_texts], dtype=torch.long)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    print({x_train.shape})
    print({y_train.shape})
    print({x_test.shape})
    print({y_test.shape})

if __name__ == "__main__":
    main()

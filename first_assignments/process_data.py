from torchtext.legacy import data 

mytokenize = lambda x : x.split()

TEXT = data.Field(sequential = True, tokenize = mytokenize,use_vocab = True,batch_first = True, fix_length = 200)

LABEL = data.Field(sequential = False, use_vocab = False, pad_token = None, unk_token = None)

text_data_fields = [ ("label", LABEL) , ("text", TEXT)]

train_data, test_data = data.TabularDataset.splits(path = "first_assignments/csvtextdata", train = "train.csv", test = "test.csv", format = "csv", skip_header = True , fields=text_data_fields)
print(len(train_data),len(test_data))

TEXT.build_vocab(train_data, max_size = 10000, vectors = None)

train_iter = data.BucketIterator(train_data, batch_size = 4)
test_iter = data.BucketIterator(test_data, batch_size = 4)  
for step, batch in enumerate(train_iter):
    if step > 0:
        break
print(batch.label)
print(batch.text.shape)  
'''
Generation of news titles using LSTM
Data taken from: https://www.kaggle.com/datasets/rmisra/news-category-dataset/
Student: Francesco Gnan
'''
# Packages
import pandas as pd
import pickle
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import string
import re
import random
from matplotlib import pyplot as plt


torch.manual_seed(42)
#########################################################

# =========== 1.1.1 Download data
dataset_name = '/kaggle/input/news-dataset/News_Category_Dataset_v3.json'
data = pd.read_json(dataset_name, lines=True)
#data = pd.DataFrame(data) 

print(data.info())
print('--------------------------------------')
headlines = data['headline']
print("Headlines:")
print(headlines)
print('--------------------------------------')
categories = data['category']
print("Categories:")
print(categories)
print('--------------------------------------')
politics_data = data[data['category'] == 'POLITICS']
print(f"POLITICS category:\n\n number of headlines: {len(politics_data)}\n")
print('\n Some examples:')
#print(politics_data.head(3))
print(politics_data.iloc[:3]['headline'])
#########################################################
# =========== 1.1.2 Tokenization
# reference: Exercise 4, split_to_names

def tokenization(sequence):
 
    EOS = "<EOS>"
    data = []

    for i in range(len(sequence)):
        text = sequence[i].lower().split()
        text += [EOS]
        data.append(text)

    return data

# reset the indeces before applying tokenization
politics_data = politics_data.reset_index(drop=True)
titles = politics_data['headline']
data_in_char = tokenization(titles)

with open('data_char.pickle', 'wb') as handle:
    pickle.dump(data_in_char, handle)

with open('data_char.pickle', 'rb') as handle:
    load_data_char = pickle.load(handle)

print('Successful save and load? ',data_in_char == load_data_char)
print(data_in_char[:3])
#########################################################
# =========== 1.1.3 Dictiionaries
# reference for the use of re: 
# https://builtin.com/software-engineering-perspectives/python-remove-character-from-string#

all_words = []
all_words.append("<EOS>")
for title in data_in_char:
    for word in title:
        #cleaned = re.sub("[^A-Za-z]","",word)
        #if cleaned not in all_words:
        if word not in all_words:
            all_words.append(word)

all_words.append("PAD")
print(len(all_words))

word_to_int = {w:i for i,w in enumerate(all_words)}
int_to_word = {i:w for w,i in word_to_int.items()}

# still 1.1.3
with open('word_to_ix.pickle', 'wb') as handle:
    pickle.dump(word_to_int, handle)
    
with open('ix_to_word.pickle', 'wb') as handle:
    pickle.dump(int_to_word, handle)

histo = {}
for title in titles:
    text = title.lower().split(' ')
    for word in text:
        #w = re.sub("[^A-Za-z]","",word)
        if word in histo:
            histo[word] += 1
        else:
            histo[word] = 1

most_common = sorted(histo, key=lambda x: histo[x], reverse=True)[:5]
for i in range(len(most_common)):
    print(most_common[i], histo[most_common[i]])
#########################################################
# =========== 1.1.4 Dataset class

def keys_to_values(keys, map):
    return [map.get(key) for key in keys]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_as_str, map):
        self.data_as_int = []

        # Convert characters to integers
        for seq_as_str in data_as_str:
            seq_as_int = keys_to_values(seq_as_str, map)

            self.data_as_int.append(seq_as_int)

    def __len__(self):
        return len(self.data_as_int)

    def __getitem__(self, ix):
        # Get data sample at index ix
        item = self.data_as_int[ix]

        # Slice x and y from sample
        x = item[:-1]
        y = item[ 1:]
        #print(x)
        #print(y)
        return torch.tensor(x), torch.tensor(y)
#########################################################
# =========== 1.1.5 Padding, Batches, Dataloader
# reference Exercise 4

def collate_fn(batch, pad_value):
  data, targets = zip(*batch)

  padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value=pad_value)
  padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True,padding_value=pad_value)

  return padded_data, padded_targets

batch_size = 64
dataset = Dataset(data_in_char, word_to_int)
if batch_size == 1:
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
else:
  dataloader = DataLoader(dataset, batch_size=batch_size,
                          collate_fn=lambda b: collate_fn(b, word_to_int["PAD"]),
                          shuffle=True)
#########################################################
# =========== 1.2 Model definition

class Model(nn.Module):
    def __init__(self, map, hidden_size, emb_dim=8, n_layers=1):
        super(Model, self).__init__()

        self.vocab_size  = len(map)
        self.hidden_size = hidden_size
        self.emb_dim     = emb_dim
        self.n_layers    = n_layers
        #self.dropout_p   = dropout_p

        # dimensions: batches x seq_length x emb_dim
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim =self.emb_dim,
            padding_idx=map["PAD"])
        
        
        self.lstm = nn.LSTM(input_size = self.emb_dim,
                            hidden_size = self.hidden_size,
                            num_layers= self.n_layers,
                            batch_first=True,
                            dropout=0.0)
        
     
        self.fc = nn.Linear(
            in_features =self.hidden_size,
            out_features=self.vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        yhat, state = self.lstm(embed, prev_state) 

        #yhat = self.dropout(yhat)
        out = self.fc(yhat)
        return out, state

    def init_state(self, b_size=1):
        
        #return torch.zeros(self.n_layers, b_size, self.hidden_size).to(DEVICE)
        h0 = torch.zeros(self.n_layers, b_size, self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.n_layers, b_size, self.hidden_size).to(DEVICE)
        return h0, c0
#########################################################
# =========== 1.3 Evaluation - part 1

def random_sample_next(model, x, prev_state, topk=5, uniform=True):
    x = x.to(model.embedding.weight.device)  # Move the input tensor to the same device as the embedding weight

    # Perform forward-prop and get the output of the last time-step
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]    # vocabulary values of last element of sequence

    # Get the top-k indexes and their values
    topk = topk if topk else last_out.shape[0]
    top_logit, top_ix = torch.topk(last_out, k=topk, dim=-1)

    # Get the softmax of the topk's and sample
    p = None if uniform else F.softmax(top_logit.detach(), dim=-1).cpu().numpy()
    sampled_ix = np.random.choice(top_ix.cpu().numpy(), p=p)

    return sampled_ix, state

def sample_argmax(model, x, prev_state, topk=5, uniform=True):
    x = x.to(model.embedding.weight.device)  # Move the input tensor to the same device as the embedding weight

    # Perform forward-prop and get the output of the last time-step
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]    # vocabulary values of the last element of the sequence

    sampled_ix = torch.argmax(last_out).item()

    return sampled_ix, state

def sample(model, seed, topk=5, uniform=True, max_seqlen=18, stop_on=None, strategy='argmax'):
    seed = seed if isinstance(seed, (list, tuple)) else [seed]

    model.eval()
    with torch.no_grad():
        sampled_ix_list = seed[:]
        x = torch.tensor([seed])

        prev_state = model.init_state(b_size=1)
        for t in range(max_seqlen - len(seed)):
            if strategy == 'argmax':
                sampled_ix, prev_state = sample_argmax(model, x, prev_state, topk, uniform)
            elif strategy == 'random':
                sampled_ix, prev_state = random_sample_next(model, x, prev_state, topk, uniform)
                
            sampled_ix_list.append(sampled_ix)
            x = torch.tensor([[sampled_ix]])

            if sampled_ix==stop_on:
                break

    model.train()
    return sampled_ix_list
#########################################################
# =========== 1.4 Training

net = Model(word_to_int, 1024, 150, n_layers=1)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_int["PAD"])
learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
n_epochs = 12

# -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:")
print(DEVICE)

net = net.to(DEVICE)
# -----------------

def write(sequence):
    result = ''
    for word in sequence:
        if word == '<EOS>':
            break
        result += word + ' '
    return result
# -----------------
def train(max_epochs, model, dataloader, criterion, optimizer, clip=1):

    costs = []
    running_loss = 0
    loss_hist = []
    perplexity = []
    
    print_every=1
    epoch = 0
    
    three_sentences = []

    while epoch < max_epochs:
        epoch += 1
        model.train()

        for input, output in dataloader:

            input = input.long()
            output = output.long()

            input = input.to(DEVICE)
            output = output.to(DEVICE)
            
            optimizer.zero_grad()
            # Initialise model's state and perform forward-prop
            prev_state = model.init_state(b_size=input.shape[0])
            out, state = model(input, prev_state)         # out has dim: batch x seq_length x vocab_size
            
            #output = output.long()

            # Calculate loss
            loss = criterion(out.transpose(1, 2), output)  #transpose is required to obtain batch x vocab_size x seq_length
            costs.append(loss.item())
            running_loss += loss.item()

            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        if print_every and (epoch%print_every)==0:
          loss_average = running_loss/float(print_every*len(dataloader))
          perplexity.append(np.exp(loss_average))
            
          print("Epoch: {}/{}, Loss: {:8.4f}, Perplexity: {:8.4f}".format(
                    int(epoch), int(max_epochs),
                    loss_average, np.exp(loss_average)))
          
          loss_hist.append(loss_average)
          running_loss = 0
  
# -------------------------------------------- 
        seed = random.choice(list(word_to_int.values())[1:-1])

        model.eval()
        
        # TODO prompt a sentence from the model
        sentence = keys_to_values(sample(model,seed, 5, False, 30, word_to_int["<EOS>"],'argmax'),int_to_word)
        print(write(sentence))
        print('----------------')
        
        if epoch == 1 or epoch == max_epochs/2 or epoch == max_epochs:
            three_sentences.append(write(sentence))
            
    return model, loss_hist, perplexity, costs, three_sentences

# --------------------------------------------
net, losses, perplexities, costs, sentences = train(n_epochs, net, dataloader, criterion, optimizer)

# -----------------
x = np.arange(len(losses))

plt.title('Loss average')
plt.plot(x, losses, color='k', label='Loss')
plt.axhline(y=1.5, color='red', linestyle='--', label='Loss = 1.5')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_average1.pdf")
plt.show()
# -----------------
x = np.arange(len(costs))

plt.title('Loss at each iteration')
plt.plot(x, costs, color='k', label='Loss')
plt.axhline(y=1.5, color='red', linestyle='--', label='Loss = 1.5')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_iter1.pdf")
plt.show()
# -----------------
x = np.arange(len(perplexities))

plt.title('Perplexity')
plt.plot(x, perplexities, color='k', label='perplexity')
plt.xlabel('Epochs')
plt.ylabel('perplexity')
plt.legend()
plt.savefig("perplexity1.pdf")
plt.show()
# -----------------

ep = ['1','6','12']
max_epochs = '12'
loss = [losses[0], losses[5], losses[-1]]
perpl = [perplexities[0],perplexities[5], perplexities[-1]]

for i in range(len(sentences)):
        
    print("Epoch: {}/{}, Loss: {:8.4f}, Perplexity: {:8.4f}".format(
                    int(ep[i]), int(max_epochs),
                    loss[i], perpl[i]))
    print('\n',sentences[i])
    print('--------------------------------')

# -------------------------------------------------
def train_with_TBBTT(max_epochs, model, dataloader, criterion, optimizer, chunk_size, device, clip=None):
    losses = []
    perplexities = []
    
    running_loss = 0
    three_sentences = []
    epoch = 0
    
    while epoch < max_epochs:
        epoch += 1
        model.train()
        for input, output in dataloader:
            # Get the number of chunks
            n_chunks = input.shape[1] // chunk_size

            # Loop on chunks
            for j in range(n_chunks):
                # TODO what is missing here?
                # Switch between the chunks
                if j < n_chunks - 1:
                    input_chunk = input[:, j * chunk_size:(j + 1) * chunk_size].to(device).to(torch.int64)
                    output_chunk = output[:, j * chunk_size:(j + 1) * chunk_size].to(device).to(torch.int64)
                else:
                    input_chunk = input[:, j * chunk_size:].to(device).to(torch.int64)
                    output_chunk = output[:, j * chunk_size:].to(device).to(torch.int64)
                # Initialise model's state and perform forward pass
                # If it is the first chunk, initialise the state to 0
                if j == 0:
                    h, c = model.init_state(b_size=input.shape[0])
                else:  # Initialize the state to the previous state - detached!
                    h, c = h.detach(), c.detach()
                    

                # Forward step
                # TODO: complete the forward step
                prev_state = (h,c)
                out, state = model(input_chunk, prev_state) 

                # Calculate loss
                # TODO complete the loss calculation
                loss = criterion(out.transpose(1, 2), output_chunk) 

                # Calculate gradients and update parameters
                # TODO: complete the backward step
                optimizer.zero_grad()
                
                running_loss += loss.item()
                loss.backward()

                # Clipping if needed
                # TODO: complete the clipping step
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                # Update parameters
                # TODO: complete the update step
                optimizer.step()

        # Print loss and perplexity every epoch
        # TODO: complete the perplexity calculation and loss / perpl priting
        if j == n_chunks - 1:
            loss_average = running_loss/float(len(dataloader)*n_chunks)
            perplexities.append(np.exp(loss_average))
            
            print("Epoch: {}/{}, Loss: {:8.4f}, Perplexity: {:8.4f}".format(
                  int(epoch), int(max_epochs), loss_average, np.exp(loss_average)))
          
            losses.append(loss_average)
            running_loss = 0
  
# -------------------------------------------- 

        model.eval()
        # TODO prompt a sentence from the model
        seed = random.choice(list(word_to_int.values())[1:-1])
   
        sentence = keys_to_values(sample(model,seed, 5, False, 30, word_to_int["<EOS>"],'argmax'),int_to_word)
        print(write(sentence))
        print('----------------')
        
        if epoch == 1 or epoch == max_epochs/2 or epoch == max_epochs-1:
            three_sentences.append(write(sentence))
            
            
    return model, losses, perplexities, three_sentences

# -----------------
net2 = Model(word_to_int, 2048, 150, n_layers=2)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_int["PAD"])
learning_rate = 0.001
optimizer = optim.Adam(net2.parameters(), lr=learning_rate)
n_epochs = 10
chunk_size = 12
device = DEVICE
net2 = net2.to(DEVICE)

# -----------------
net2, losses, perplexities, sentences = train_with_TBBTT(n_epochs, net2, dataloader, criterion, optimizer,chunk_size, device, 1)
# -----------------
x = np.arange(len(losses))

plt.title('Training Loss, TBBTT')
plt.plot(x, losses, color='k', label='Loss')
plt.axhline(y=1, color='red', linestyle='--', label='Loss = 1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss2.pdf")
plt.show()
# -----------------
x = np.arange(len(perplexities))

plt.title('Perplexity, TBBTT')
plt.plot(x, perplexities, color='k', label='Perplexity')
plt.xlabel('Iterations')
plt.ylabel('perplexity')
plt.legend()
plt.savefig("perplexity2.pdf")
plt.show()
# -----------------
ep = ['1','5','10']
max_epochs = '10'
loss = [losses[0], losses[4], losses[-1]]
perpl = [perplexities[0],perplexities[4], perplexities[-1]]

for i in range(len(sentences)):
        
    print("Epoch: {}/{}, Loss: {:8.4f}, Perplexity: {:8.4f}".format(
                    int(ep[i]), int(max_epochs),
                    loss[i], perpl[i]))
    print('\n',sentences[i])
    print('--------------------------------')

#########################################################
# =========== 1.5 Evaluation - part 2

def generate_sentence( prompt, strategy ):

    net2.eval()
    prompt = list(word_to_int[word] for word in prompt.split())

    
    if strategy == 'sample':
        sentence = keys_to_values(sample(net2,prompt, 5, False, 30, word_to_int["<EOS>"],'argmax'),int_to_word)
    elif strategy == 'greedy':
        sentence = keys_to_values(sample(net2,prompt, 5, False, 30, word_to_int["<EOS>"],'random'),int_to_word)

    return write(sentence)
    
prompt = 'the president wants'

for i in range(3):
    sent = generate_sentence( prompt, 'sample' )
    print('\n',sent)
    
print('\n ---------------- \n')
for i in range(3):
    sent = generate_sentence( prompt, 'greedy' )
    print('\n',sent)

#########################################################
# =========== BONUS
       
from sklearn.metrics.pairwise import cosine_similarity


def get_embedding_vector(word, model_instance, word_to_int):
    if word in word_to_int:
        
        word_index = word_to_int[word]
        word_tensor = torch.tensor([[word_index]])
        word_tensor = word_tensor.to(model_instance.embedding.weight.device)
        
        embedding_vector = model_instance.embedding(word_tensor)
        embedding_vector = embedding_vector.squeeze().detach().cpu().numpy()
        
        return embedding_vector



def find_most_similar_words(target_embedding, model_instance, int_to_word, top_n=10):

    all_embeddings = model_instance.embedding.weight.detach().cpu().numpy()
    similarities = cosine_similarity([target_embedding], all_embeddings)[0]
    
    top_indices = similarities.argsort()[-top_n:][::-1]
    similar_words = [int_to_word[idx] for idx in top_indices]
    
    return similar_words

word1 = "king"
word2 = "man"
embedding_vector1 = get_embedding_vector(word1, net, word_to_int)
embedding_vector2 = get_embedding_vector(word2, net, word_to_int)

diff = embedding_vector1 - embedding_vector2 
similar_words = find_most_similar_words(diff, net, int_to_word)

print(f"Most similar words to '{word1}' - '{word2}': {similar_words}")
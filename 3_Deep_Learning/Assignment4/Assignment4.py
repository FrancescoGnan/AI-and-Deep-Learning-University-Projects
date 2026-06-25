'''
Template for the 4th assignment
Student: NAME SURNAME
'''

############################
# Packages
############################
import torch
import torch.nn as nn
import math
import string
import regex as re
import matplotlib.pyplot as plt
import pickle
import random
from torch.utils.data import DataLoader
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

############################
# Classes
############################
# Vocabulary class
class Vocabulary:
    '''
    Class for dealing with our corpus
    '''

    def __init__(self, name, pairs):
        """
        Args:
            name (str): name of the language
            pairs (list): list of pairs of sentences
        """
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.pairs = pairs
        
    def __len__(self):
        """
        Return the size of the vocabulary.
        """
        return len(self.word2index)


    def add_word(self, word):
        '''
        Add a word to the vocabulary
        :param word: a string
        '''
        # TODO: add the word to the vocabulary
        if word and word not in self.word2index:
            idx = len(self.word2index)
            self.word2index[word] = idx
            self.index2word[idx] = word
        
        #if word not in self.word2index:

         #   idx = len(self.word2index)
            
         #   self.word2index[word] = idx
          #  self.index2word[idx] = word

    def add_sentence(self, sentence):
        '''
        Add a sentence to the vocabulary
        :param sentence: list of strings (words)
        '''
        # TODO add the sentence to the vocabulary, this method will call the add_word method
        for word in sentence:
            #if word not in self.word2index:
                self.add_word(word)
                

def clear_punctuation(s):
        '''
        This function removes all the punctuation from a sentence and insert a blank between any letter and !?.
        :param s: a string
        :return: the "cleaned" string
        '''
        re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # Remove all the character that are not letters, puntuation or numbers
        # Insert a blank between any letter and !?. using regex
        s = re.sub(r"([a-zA-Z])([!?.])", r"\1 \2", s)
        return s

# Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocabulary, pairs):
        # TODO We want vocabulary and pairs to be attributes of the class
        self.vocab = vocabulary
        self.pairs = pairs

    def __len__(self):
        # TODO how many pairs do we have?
        return len(self.pairs)

    def __getitem__(self, ix):
        # TODO returns two tensors (question, answer) of the pair at index ix
        # TODO the tensors should be of type torch.tensor and should contain integers (word indices)
       
        #try:
            pair = self.pairs[ix]
            question_ix = []
            answer_ix = []
            question_word = []
            answer_word = []

            for word in pair:
                question_word.append(word)
                if word == '<EOS>':
                    break
                    
            answer_word = pair[len(question_word):]

            #print('answer_word', answer_word)
            question_ix = [self.vocab.word2index[word] for word in question_word]
            answer_ix = [self.vocab.word2index[word] for word in answer_word]    
            #print('answer_ix',answer_ix)
            #print('question_ix', question_ix)
            #print('answer_ix', answer_ix)

            question = torch.tensor(question_ix, dtype=torch.long)
            answer = torch.tensor(answer_ix, dtype=torch.long)

            return question, answer
        
        #except Exception as e:
            #print(f"Error at index {ix}: {e}")
            #return None, None
    
class PositionalEncoding(nn.Module):
    '''
    Adapted from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Positional encoding is added to the input embeddings to give the model 
    some information about the relative or absolute position of the tokens in the sequence. 
    
    The constructor initializes the PositionalEncoding module. 
    It takes the dimension of the model (d_model), dropout rate (dropout),
    and the maximum length of the sequence (max_len) as parameters. 
    It creates a positional encoding matrix (pe) using sine and cosine functions.
    
    The forward method adds the positional encoding to the input tensor x.
    It first checks if the length of the sequence is less than the specified 
    max_len to avoid indexing errors. Then, it adds the positional encoding to the input tensor and applies dropout.
    '''
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        try:
            assert x.size(0) < self.max_len
        except:
            print("The length of the sequence is bigger than the max_len of the positional encoding. Increase the max_len or provide a shorter sequence.")
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, pad_id=0, encoder_layers=6, decoder_layers=6, dim_feedforward=2048, num_heads=8, dropout_p=0.1):
        super().__init__()
        # TODO add an embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # TODO add a positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, dropout_p)
        # TODO add a transformer layer, you can use nn.Transformer. You can use the default values for the parameters, but what about batch_first?
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            batch_first=True  
        )
        #batch_first parameter is a boolean argument that determines the ordering 
        #of the input tensors. Specifically, it defines whether the first dimension 
        #of the input tensors represents the batch size or the sequence length.
        #If batch_first is set to True, the input tensors are expected to have the shape 
        #(batch_size, sequence_length, features). This means that the batch size is the first 
        #dimension of the input tensor. If batch_first is set to False (the default), 
        #the input tensors are expected to have the shape (sequence_length, batch_size, features), 
        #where the sequence length is the first dimension.


   
        # TODO add a linear layer. Note: output should be probability distribution over the vocabulary
        self.linear = nn.Linear(d_model, vocab_size)


        # Stuff you may need
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.num_heads = num_heads

    def create_padding_mask(self, x, pad_id=0):
        # TODO create a boolean mask for the <PAD> tokens
        padding_mask = (x == pad_id)
        return padding_mask

        #element-wise equality check between the input sequence x 
        #and the specified pad_id. The result is a boolean tensor 
        #where each element is True if the corresponding element in 
        #x is equal to pad_id, and False otherwise.

    def forward(self, src, tgt):
        # S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        # src: (N, S)
        # tgt: (N, T)
        # src_pad_mask: (N, S)
        # tgt_pad_mask: (N, T)
        # mask the future : (N * num_heads, T, T)
        src_pad_mask = self.create_padding_mask(src, self.pad_id) # (N, S)
        tgt_pad_mask = self.create_padding_mask(tgt, self.pad_id) # (N, T)

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.pos_encoder(src)  # (N, S, E)
        tgt = self.pos_encoder(tgt) # (N, T, E)

        # Mask the memory
        memory_key_padding_mask = src_pad_mask  # (N, S)

        # Mask the future
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)  # (T, T)
        #tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1), dtype=torch.bool).to(tgt.device) # (T, T)
        # Expand to make it N * num_heads, T, T
        
        tgt_mask = tgt_mask.unsqueeze(0).repeat(tgt.size(0) * self.num_heads, 1, 1) # (N, T, T)
        
        #print("Shapes:")
        #print("src:", src.shape)
        #print("tgt:", tgt.shape)
        #print("src_pad_mask:", src_pad_mask.shape)
        #print("tgt_pad_mask:", tgt_pad_mask.shape)
        #print("memory_key_padding_mask:", memory_key_padding_mask.shape)
        #print("tgt_mask:", tgt_mask.shape)
        # Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=memory_key_padding_mask) # (N, T, E)
        # Linear layer
        output = self.linear(output) # (N, T, V)
        return output

    
    
    
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device="cuda"):
        
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0

            for idx, (src, tgt) in enumerate(train_loader):

                src, tgt = src.to(device), tgt.to(device)

                src = src.long()
                tgt = tgt.long()
                
                optimizer.zero_grad()

                # Forward pass
                output = model(src, tgt[:, :-1])
                target = tgt[:, 1:]
                loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))

                # Backward pass
                loss.backward()
                #to avoid exploding gradients I apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_train_loss += loss.item()
  

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            total_val_loss = 0.0

            #with torch.no_grad():
            for val_src, val_tgt in val_loader:
                    val_src, val_tgt = val_src.to(device), val_tgt.to(device)

                    val_output = model(val_src, val_tgt[:, :-1])
                    val_target = val_tgt[:, 1:]
                    val_loss = criterion(val_output.reshape(-1, val_output.size(-1)), val_target.reshape(-1))

                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Learning rate scheduling
            scheduler.step()

        return train_losses, val_losses


def train_ga(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda',accumulation_steps=32):
    train_losses = []
    val_losses = []
    accumulation_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            src = src.long()
            tgt = tgt.long()

            optimizer.zero_grad()

            # Forward pass
            output = model(src, tgt[:, :-1])
            target = tgt[:, 1:]
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))

            # Backward pass
            loss.backward()

            accumulation_counter += 1

            # Gradient accumulation
            if accumulation_counter % accumulation_steps == 0:
                optimizer.step()
                accumulation_counter = 0

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        #with torch.no_grad():

        for val_src, val_tgt in val_loader:
                val_src, val_tgt = val_src.to(device), val_tgt.to(device)
                val_output = model(val_src, val_tgt[:, :-1])
                val_target = val_tgt[:, 1:]
                val_loss = criterion(val_output.reshape(-1, val_output.size(-1)), val_target.reshape(-1))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step()


    return train_losses, val_losses


############################
# Methods
############################

if __name__ == "__main__":
    # !!! Don't change the seed !!!
    torch.manual_seed(42)
    # !!!!!!
    
    
    #####################################
    #  1. DATA
    ####################################
    
    # --------------- 1,1 Download the data
    conversations= '/kaggle/input/datasett/Data/movie_conversations.txt'
    lines = '/kaggle/input/datasett/Data/movie_lines.txt'

    def read_and_print(file_path, num_lines=5):
        with open(file_path, 'r') as file:
            for _ in range(num_lines):
                print(file.readline())

    print("Movie Conversations:")
    read_and_print(conversations)

    print("\nMovie Lines:")
    read_and_print(lines)
    
    # --------------- 1.2 Create the pairs
    
    def get_pairs(file):
        pairs = []
        i = 0
        a = []
        for line in file:
            components = line.strip().split(' +++$+++ ')
            a.append(components[-1])
            
            if i == 1:
                pairs.append(a)
                i = 0
            else:
                i += 1
            
        return pairs
    
    
    def get_first_two_sentences(file):
        sentence_pairs = []

        for line in file:
            components = line.strip().split(' +++$+++ ')
            sentence = components[-1]

            # Check if sentence_pairs is not empty and the last pair has less than two sentences
            if sentence_pairs and len(sentence_pairs[-1]) < 2:
                sentence_pairs[-1].append(sentence)
            else:
                sentence_pairs.append([sentence])

        # Filter out incomplete pairs
        sentence_pairs = [pair for pair in sentence_pairs if len(pair) == 2]

        return sentence_pairs


    
    list_of_pairs = []
    
    file = open(lines, 'r',encoding='latin-1', errors='ignore')
    list_of_pairs = get_first_two_sentences(file)
    print('-------------')
    print(list_of_pairs[:2])

    # --------------- 1.3 Tokenize the data
    def tokenization(pairs):
 
        EOS = "<EOS>"
        SOS = "<SOS>"
        data = []
        punctuation_symbols = {'?', '.', '!'}

        for pair in pairs:
            combined_tokens = []  # Lista per unire le due frasi della coppia

            for index, sentence in enumerate(pair):                
                tokens = []
                for word in sentence.lower().split():
                    # Check if the word ends with a punctuation symbol
                    if word[-1] in punctuation_symbols:
                        # Split the word and add the punctuation as a separate token
                        tokens.extend([word[:-1], word[-1]])
                    else:
                        tokens.append(word)
                tokens.append(EOS)
                if index == 1:
                    tokens.insert(0, SOS)
                combined_tokens.extend(tokens)  # Unisci i tokens della frase corrente alla lista combinata

            data.append(combined_tokens)
        return data
    
    def tokenization_2(pairs):
        EOS = "<EOS>"
        SOS = "<SOS>"
        data = []

        for pair in pairs:
            combined_tokens = []  # Lista per unire le due frasi della coppia

            for index, sentence in enumerate(pair):
                cleaned = clear_punctuation(sentence)
                tokens = []
                for word in cleaned.lower().split():
                   tokens.append(word)
                tokens.append(EOS)
                if index == 1:
                    tokens.insert(0, SOS)
                combined_tokens.extend(tokens)  # Unisci i tokens della frase corrente alla lista combinata

            data.append(combined_tokens)
        return data
    
    tokenized_pairs = tokenization(list_of_pairs)
    print(tokenized_pairs[:2])
        
    tok2 = tokenization_2(list_of_pairs)
    print(tok2[:2])
    
    # --------------- 1.4 Remove pairs

    def length_distribution(pairs):
        

        distribution = {}

        for pair in pairs:
            first_eos_index = pair.index('<EOS>')
            second_sos_index = pair.index('<SOS>', first_eos_index + 1)
            second_eos_index = pair.index('<EOS>', second_sos_index)
            
            strings_before_first_eos = first_eos_index
            strings_between_sos_eos = second_eos_index - second_sos_index - 1  # Exclude <SOS>
            
            two_lenghts = [strings_before_first_eos, strings_between_sos_eos]
            
            for length in two_lenghts:
                if length in distribution:
                    distribution[length] += 1
                else:
                    distribution[length] = 1

        return distribution

    
    distr = length_distribution(tok2)
    #distr = length_distribution(tokenized_pairs)

    keys = list(distr.keys())
    values = list(distr.values())

    plt.bar(keys, values, color='blue')
    plt.xlabel('Number of words')
    plt.ylabel('Frequency')
    plt.title('Lengths distribution')
    plt.xlim(0,50)
    plt.axvline(x=25.5, color='red', linestyle='--', linewidth=2)
    plt.savefig("bar_length.pdf")

    plt.show()
    
    def remove_pairs(pairs, max_length=25):
        filtered_pairs = []

        for pair in pairs:
            first_eos_index = pair.index('<EOS>')
            second_sos_index = pair.index('<SOS>', first_eos_index + 1)
            second_eos_index = pair.index('<EOS>', second_sos_index)
            
            strings_before_first_eos = first_eos_index
            strings_between_sos_eos = second_eos_index - second_sos_index - 1  # Exclude <SOS>
            
            two_lenghts = [strings_before_first_eos, strings_between_sos_eos]
            
            if two_lenghts[0] > max_length or two_lenghts[1] > max_length:
                continue
            else:
                filtered_pairs.append(pair)
                
        return filtered_pairs

    print(f'---------------\n tok2 size before removing: {len(tok2)}')
    new_list = remove_pairs(tok2)
    print(f'tok2 size after removing: {len(new_list)}')

    with open('cleaned_pairs.pickle', 'wb') as handle:
        pickle.dump(new_list, handle)
    


    # --------------- 1.6 Remove rare
    
    def words_frequency(pairs, min_frequency=50):
        histo = {}
        for pair in pairs:
            for word in pair:
                if word in histo:
                    histo[word] += 1
                else:
                    histo[word] = 1
        
        filtered_histo = {word: freq for word, freq in histo.items() if freq >= min_frequency}
            
        return filtered_histo
    

    word_hist = words_frequency(new_list)
    
    keys = list(word_hist.keys())
    values = list(word_hist.values())

    plt.bar(keys, values, color='blue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Words frequency')
    plt.ylim(0,50000)
    #plt.axvline(x=15.5, color='red', linestyle='--', linewidth=2)
    plt.xticks([])
    plt.savefig("rare_words.pdf")

    plt.show()
    
    num_bins = int(np.sqrt(len(values)))+700

    plt.hist(values, bins=num_bins, edgecolor='black', color='blue')
    plt.xlabel('Frequency')
    plt.ylabel('Number of Words')
    plt.title('Distribution of Word Frequencies')
    plt.xlim(0, 17500)

    # Remove x-axis labels
    #plt.xticks([])

    plt.show()
    

    def delete_words(pairs, histo, threshold=50):
        filtered_pairs = []

        for pair in pairs:
            if all(word in histo and histo[word] >= threshold for word in pair):
                filtered_pairs.append(pair)

        return filtered_pairs

    # Set a suitable threshold based on the distribution (adjust as needed)
    #threshold = 5
    filtered_pairs = delete_words(new_list, word_hist)

    print(f'Pairs count before eliminating words: {len(new_list)}')
    print(f'Pairs count after eliminating words: {len(filtered_pairs)}')

    # Save filtered pairs to a new pickle file
    with open('filtered_pairs.pickle', 'wb') as handle:
        pickle.dump(filtered_pairs, handle)
    
    #print('filtered_pairs', filtered_pairs)
    
    # --------------- 1.8 Sample subset

    random.seed(42)
    sampled_sentences = random.sample(new_list, 20000)
    
    total_samples = len(sampled_sentences)
    split_ratio = 0.7
    split_index = int(split_ratio * total_samples)

    random.shuffle(sampled_sentences)

    train_data = sampled_sentences[:split_index]
    val_data = sampled_sentences[split_index:]
    
    # --------------- 1.9 Sample subset

    vocab = Vocabulary('conversations', sampled_sentences)
    for sentence in sampled_sentences:
        vocab.add_sentence(sentence)
        
    train_vocab = Vocabulary('train_conversations', train_data)
    for sentence in train_data:
        train_vocab.add_sentence(sentence)

    val_vocab = Vocabulary('val_conversations', val_data)
    for sentence in val_data:
        val_vocab.add_sentence(sentence)

        

    def collate_fn(batch, pad_value):
        data, targets = zip(*batch)
    
        padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value=pad_value)
        padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True,padding_value=pad_value)

        return padded_data, padded_targets


    batch_size = 32
    train_dataset = Dataset(train_vocab, train_data)
    val_dataset = Dataset(val_vocab, val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,\
                                collate_fn=lambda b: collate_fn(b, train_vocab.word2index["<PAD>"]))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,\
                            collate_fn=lambda b: collate_fn(b, val_vocab.word2index["<PAD>"]))
    

   
    #dataset = Dataset(sampled_sentences, vocab.word2index)
    #if batch_size == 1:
    #  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #else:
    #  dataloader = DataLoader(dataset, batch_size=batch_size,
    #                          collate_fn=lambda b: collate_fn(b, vocab.word2index["PAD"]),
    #                          shuffle=True)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ',device)

    ntokens = len(vocab)  # size of vocabulary
    d_model = 512 
    pad_id = 0
    encoder_layers = 10
    decoder_layers = 10
    dim_feedforward = 2048
    nhead = 8 
    dropout = 0.25
    model1 = TransformerModel(ntokens, d_model, pad_id, encoder_layers, decoder_layers, dim_feedforward, nhead, dropout).to(device)
    
    criterion = nn.CrossEntropyLoss()
    lr = 0.0005
    optimizer = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    
    print('\n ----------------------')
    print('\n Start training:\n')


    train_losses1, val_losses1 = train(model1, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

    
    
    x = np.arange(len(train_losses1))

    plt.title('loss')
    plt.plot(x, train_losses1, label='train')
    plt.plot(x, val_losses1, label='validation')
    plt.legend()
    plt.savefig("loss1.pdf")

    plt.show()
    
    #gradient accumulation
    ntokens = len(vocab)  # size of vocabulary
    d_model = 512 
    pad_id = 0
    encoder_layers = 6
    decoder_layers = 6
    dim_feedforward = 2048
    nhead = 8 
    dropout = 0.3
    model2 = TransformerModel(ntokens, d_model, pad_id, encoder_layers, decoder_layers, dim_feedforward, nhead, dropout).to(device)
    
    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    original_batch_size = 2
    accumulation_steps = 32

    # Modify the data loaders
    effective_batch_size = original_batch_size * accumulation_steps

    # Assuming train_loader and val_loader are your original data loaders
    # Update batch size to the effective batch size
    train_loader.dataset.batch_size = effective_batch_size
    val_loader.dataset.batch_size = effective_batch_size

    # Call the train_ga function
    train_losses2, val_losses2 = train_ga(
        model2,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=10,
        device="cuda",
        accumulation_steps=accumulation_steps
    )

    x = np.arange(len(train_losses2))

    plt.title('loss')
    plt.plot(x, train_losses2, label='train')
    plt.plot(x, val_losses2, label='validation')
    plt.legend()
    plt.savefig("loss2.pdf")

    plt.show()
    
    def generate_answer_greedy(model, vocab, input_sentence, max_length=50, device="cuda"):

        model.eval()

        input_indices = [vocab.word2index[word] for word in input_sentence]

        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
        output_tensor = torch.full((input_tensor.size(0),1), vocab.word2index["<SOS>"], dtype=torch.long).to(input_tensor.device)

        for _ in range(max_length - 1):
            output_forward = model.forward(input_tensor, output_tensor)
            probs = F.softmax(output_forward[:, -1, :], dim=-1).detach().cpu().numpy()
            
            predicted_word_index = np.argmax(probs)

            predicted_word = torch.tensor([[predicted_word_index]], dtype=torch.long).to(input_tensor.device)
            
            output_tensor = torch.cat([output_tensor, predicted_word], dim=-1)
            if predicted_word.item() == vocab.word2index["<EOS>"]:
                break
        output_tensor = output_tensor.cpu().numpy()
        generated_answer = []
        for i in range(output_tensor.shape[1]):
            generated_answer.append(vocab.index2word[output_tensor[0,i]])
        return generated_answer

    def generate_answer_topk(model, vocab, input_sentence, max_length=50, device="cuda", topk=5):

        model.eval()

        input_indices = [vocab.word2index[word] for word in input_sentence]

        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
        output_tensor = torch.full((input_tensor.size(0),1), vocab.word2index["<SOS>"], dtype=torch.long).to(input_tensor.device)

        for _ in range(max_length - 1):
            output_forward = model.forward(input_tensor, output_tensor)
            probs = F.softmax(output_forward[:, -1, :], dim=-1).detach().cpu().numpy()
            topk_index = np.argsort(probs[0])[-topk:]
            
            predicted_word_index = np.random.choice(topk_index)

            predicted_word = torch.tensor([[predicted_word_index]], dtype=torch.long).to(input_tensor.device)
            
            output_tensor = torch.cat([output_tensor, predicted_word], dim=-1)
            if predicted_word.item() == vocab.word2index["<EOS>"]:
                break
        output_tensor = output_tensor.cpu().numpy()
        generated_answer = []
        for i in range(output_tensor.shape[1]):
            generated_answer.append(vocab.index2word[output_tensor[0,i]])
        return generated_answer



    question1 = ['how','old', 'are','you', '?']
    question2 = ['what','is', 'your','name', '?']
    question3 = ['do', 'you', 'know', '?']

    greedy_answer1 = generate_answer_greedy(model1, vocab, question1, max_length=10, device="cuda")
    topk_answer1 = generate_answer_topk(model1, vocab, question1, max_length=10, device="cuda", topk=5)

    greedy_answer2 = generate_answer_greedy(model1, vocab, question2, max_length=10, device="cuda")
    topk_answer2 = generate_answer_topk(model1, vocab, question2, max_length=10, device="cuda", topk=5)

    greedy_answer3 = generate_answer_greedy(model1, vocab, question3, max_length=10, device="cuda")
    topk_answer3 = generate_answer_topk(model1, vocab, question3, max_length=10, device="cuda", topk=5)

    print('\n Question 1: ', question1)
    print('\n Greedy Answer: ', greedy_answer1)
    print('\n Topk Answer: ',topk_answer1)
    print('---------------------------')
    print('\n Question 2: ', question2)
    print('\n Greedy Answer: ', greedy_answer2)
    print('\n Topk Answer: ',topk_answer2)
    print('---------------------------')
    print('\n Question 3: ', question3)
    print('\n Greedy Answer: ', greedy_answer3)
    print('\n Topk Answer: ',topk_answer3)



# BONUS -----------------
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ',device)

    ntokens = len(vocab)  # size of vocabulary
    d_model = 512 
    pad_id = 0
    encoder_layers = 10
    decoder_layers = 10
    dim_feedforward = 2048
    nhead = 8 
    dropout = 0.25
    model3 = TransformerModel(ntokens, d_model, pad_id, encoder_layers, decoder_layers, dim_feedforward, nhead, dropout).to(device)
    
    criterion = nn.CrossEntropyLoss()
    lr = 0.0005
    optimizer3 = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=1e-4)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=2)
    model3, optimizer3, train_loader, val_loader, scheduler3 = accelerator.prepare(
        model3, optimizer3, train_loader, val_loader, scheduler3
    )

    def train_ga_hf(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda',accumulation_steps=32):
        train_losses = []
        val_losses = []
        accumulation_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0

        
            for idx, (src, tgt) in enumerate(train_loader):
                with accelerator.accumulate(model):
                    src, tgt = src.to(device), tgt.to(device)

                    src = src.long()
                    tgt = tgt.long()
                    optimizer.zero_grad()

                    output = model(src, tgt[:, :-1])
                    target = tgt[:, 1:]
                    loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))

                    accelerator.backward(loss)
                    #loss.backward()
                
                    optimizer.step()

                    total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

        
            # Validation
            model.eval()
            total_val_loss = 0.0
            #with torch.no_grad():

            for val_src, val_tgt in val_loader:
                    val_src, val_tgt = val_src.to(device), val_tgt.to(device)
                    val_output = model(val_src, val_tgt[:, :-1])
                    val_target = val_tgt[:, 1:]
                    val_loss = criterion(val_output.reshape(-1, val_output.size(-1)), val_target.reshape(-1))
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Learning rate scheduling
            scheduler.step()


        return train_losses, val_losses

    train_losses3, val_losses3 = train_ga_hf(model3, train_loader, val_loader, criterion, optimizer3, scheduler3, num_epochs=10)

    
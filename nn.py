# Author: Claudio Moises Valiense de Andrade, Licence: MIT

import claudio_funcoes as cv # functions of general use
import torch # lib for deep learning
from torch import nn # neural network in pytorch
from sklearn.feature_extraction.text import TfidfVectorizer # Apply TF-IDF
import random # utilize in random iteration
import sys # system input
import numpy # manipulate vector
from torch.nn import functional as F # function one hot encoder
import collections # count fast
import random # shurffle data
from d2l import torch as d2l
from torch.utils import data # trabalhar com dados iterados
import os # read files
import timeit # mensure time
from sklearn.datasets import dump_svmlight_file # save format svmlight
import gensim # load embedding
from sklearn.decomposition import TruncatedSVD # reduction dimension sparse

random.seed(42); torch.manual_seed(42); numpy.random.seed(seed=42) # reproducibily soluction
cont=0
first_epoch = True
last_epoch = False # state last epch
r_lstm = []

def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional recurrent neural
        # network
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        global cont
        global first_epoch
        global last_epoch
        global r_lstm
        #print(f'cont: {cont}')
        cont+=1
        # The shape of `inputs` is (batch size, no. of words). Because LSTM
        # needs to use sequence as the first dimension, the input is
        # transformed and the word feature is then extracted. The output shape
        # is (no. of words, batch size, word vector dimension).

        embeddings = self.embedding(inputs.T) # cada linha que representa um token, e convertida em 300 dimensoes, que representa um embedding de um token, defini aleatoriamente os valores das 300 dimensoes,quando faz o treinamento que a semelhança entre palavra sao ajustadas (nao usando glove/word2vec)

        #print(f'embeddings: {len(embeddings[0])}'); exit()
        # Since the input (embeddings) is the only argument passed into
        # nn.LSTM, both h_0 and c_0 default to zero.
        # we only use the hidden states of the last hidden layer
        # at different time step (outputs). The shape of `outputs` is
        # (no. of words, batch size, 2 * no. of hidden units).
        self.encoder.flatten_parameters() 
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        if last_epoch == True: # save representation
            for index_doc in range(len(encoding)):
                r_lstm.append(encoding[index_doc].cpu().detach().numpy())
            
        outs = self.decoder(encoding)
        print(outs); exit()
        #if first_epoch == True:
        #    print(f'input: {inputs}') # transforma os embeddings que estao em coluna, em linhas
        #    print(f'embeddings: {embeddings[0][0][0:20]}, len: {len(embeddings[0])}')
        #    print(f'outputs: {outputs.shape}')
        #    print(f'encoding: {encoding.shape}')
        #    first_epoch=False

        return outs

def evaluate_model(net, x_data, y_test, batch_size): 
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode
    y_pred = net( x_data.float() ).argmax(axis=1).tolist() # define class with max value
    y_test = y_test.cpu().tolist()
    return cv.f1([y_test], [y_pred])[0]
    
def evaluate_model2(net, x_test, y_test, batch_size, loss): 
    """Compute the accuracy for a model on a dataset."""
    metric_batch = 0; count_docs=0
    net.eval()  # Set the model to evaluation mode
    loss_batch =0
    
    for x_data, y_test in x_test:
        x_data = x_data.to(device='cuda'); y_test = y_test.to(device='cuda')
        count_docs += len(y_test)
        out = net( x_data, x_data, x_data, None )
        y_pred = out.argmax(axis=1).tolist() # define class with max value
        #y_pred = net( x_data ).argmax(axis=1).tolist() # define class with max value
        l = loss(out, y_test)
        loss_batch += float(l.sum()) * len(y_test) #um loss diferente para cada documento
        y_test = y_test.cpu().tolist()
        metric_batch += cv.f1([y_test], [y_pred])[0] * len(y_test)
    return loss_batch / count_docs, metric_batch / count_docs

def data_iter(batch_size, features, labels):
    """Funcao que gerar os minibatch, para utilizar no gradient descente, nao precisamos rodar o conjunto todo a cada iteracao"""
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def train_epoch(net, x_train, loss, updater, y_train, batch_size):  #@save
    """The training loop defined in Chapter 3."""
    net.train() # Set the model to training mode
    loss_batch = 0; metric_batch=0; first = True
    for X, y in  data_iter(batch_size, x_train, y_train): # Compute gradients and update parameters
        y_hat = net(X.float())
        l = loss(y_hat, y) 
        updater.zero_grad()
        l.backward()
        updater.step()
        loss_batch += (float(l) * len(y)) # calculate loss batch
        y_pred = y_hat.argmax(axis=1).tolist() # define class with max value
        y = y.cpu().tolist() # necessario ao usar GPU
        metric_batch += cv.f1([y], [y_pred])[0] * len(y) # calculate per batch metric, considerando the qtd elements
    return loss_batch/len(y_train), metric_batch/len(y_train)

def execute_model(net, train_iter, test_iter, loss, num_epochs, updater, y_train, y_test, batch_size):  #@save
    """Train a model (defined in Chapter 3)."""
    for epoch in range(num_epochs):
        train_loss, train_f1 = train_epoch(net, train_iter, loss, updater, y_train, batch_size)
        valid_f1 = evaluate_model(net, test_iter, y_test, batch_size)
        print(f'epoch: {epoch}, train_loss: {train_loss}, train_f1: {train_f1}, valid_f1: {valid_f1}')
        if train_f1 > 0.95: break # overfitting dataset debate
    return train_loss, train_f1, valid_f1

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad



def train_test_vocab(x_train, y_train, x_test, y_test,  batch_size, fix_lenght):
    """Return iterador train and test"""
    train_tokens = tokenize(x_train)
    test_tokens = tokenize(x_test)
    print(f'x:train, {len(x_train)}, x_test:  {len(x_test)}')

    print(f'x_train[0]: {x_train[0][0:10]}')
    print(f'x_train[1]: {x_train[1][0:10]}')
    print(f'ids train[0]: {train_tokens[0]}')

    vocab = cv.my_vocab(train_tokens)
    print('vocab: ', len(vocab))
    #vocab = my_vocab( all_doc(train_tokens, test_tokens) )
    #print(len(train_tokens), len(test_tokens)); exit()

    # tfidf
    lenght_reduction = 768
    vectorizer = TfidfVectorizer( ngram_range=(1,2) )
    x_train = vectorizer.fit_transform(x_train)
    print('shape', x_train.shape)
    x_test = vectorizer.transform(x_test)
    svd = TruncatedSVD(n_components=lenght_reduction, random_state=42)
    x_train = svd.fit_transform(x_train)
    print('shape', x_train.shape)
    x_test = svd.transform(x_test)

    train_glove, test_glove = cv.transform_emb(train_tokens, test_tokens)
    train_glove = [numpy.mean(tokens,axis=0) for tokens in train_glove]
    test_glove = [numpy.mean(tokens,axis=0) for tokens in test_glove]

    add_zero = 768 - len(x_train[0])
    if add_zero >1: # case svd no reduce in 768
        x_train = [numpy.append( tokens, numpy.zeros(add_zero) ) for tokens in x_train]
        x_test = [numpy.append( tokens, numpy.zeros(add_zero) ) for tokens in x_test]
    train_glove = [numpy.append( tokens, numpy.zeros(468) ) for tokens in train_glove]
    test_glove = [numpy.append( tokens, numpy.zeros(468) ) for tokens in test_glove]
    
    print(len(x_train[0]), len(train_glove[0]))

    x_train = [ numpy.stack((x_train[index_doc], train_glove[index_doc])) for index_doc in range(len(x_train))]
    x_test = [ numpy.stack((x_test[index_doc], test_glove[index_doc])) for index_doc in range(len(x_test))]


    train_features = torch.tensor( [ x for x in x_train] ).float() # x entre cochetes para representar apenas o equivalente a um token
    test_features = torch.tensor( [ x for x in x_test]).float()
    print(train_features.shape)
    #print(train_features.shape); exit()
    train_iter = load_array( ( train_features, torch.tensor(y_train) ), batch_size, False)
    test_iter =  load_array( ( test_features, torch.tensor(y_test) ), batch_size, False)
    return train_iter, test_iter, vocab
    exit()
    
    # glove
    train_features, test_features = cv.transform_emb(train_tokens, test_tokens)
    train_features = [numpy.mean(tokens,axis=0) for tokens in train_features]
    test_features = [numpy.mean(tokens,axis=0) for tokens in test_features]
    
    train_features = torch.tensor( [ [x] for x in train_features]).float()
    test_features = torch.tensor( [ [x] for x in test_features] ).float()
    #print(f'shape: {train_features.shape}'); exit()
    train_iter = load_array( ( train_features, torch.tensor(y_train) ), batch_size, False)
    test_iter =  load_array( ( test_features, torch.tensor(y_test) ), batch_size, False)
    return train_iter, test_iter, vocab

    # embedding module torch
    train_ids = text_in_id(train_tokens, vocab)
    test_ids = text_in_id(test_tokens, vocab)
    #print( [x for x in train_ids] ); exit()
    train_features = torch.tensor( [truncate_pad(x, fix_lenght, 0) for x in train_ids] )
    test_features = torch.tensor( [truncate_pad(x, fix_lenght, 0) for x in test_ids] )
    #print(f'shape: {train_features[0]}'); exit()
    #train_features=torch.tensor([train_ids]); test_features = torch.tensor([test_ids])
    #print(train_features[0]); exit() 

    #print(truncate_pad(new_x[2], 50, 0));exit()
    
    #vocab = Vocab(train_tokens)
    #print(vocab['ashjdkas']); exit()
    '''    

    #print(train_tokens[0]); print( vocab[train_tokens[0]] ); print( truncate_pad(vocab[train_tokens[0]], fix_lenght, vocab['<pad>']) ) 
    train_features = torch.tensor([truncate_pad(vocab[line], fix_lenght, vocab['<pad>']) for line in train_tokens]) # define que todos os documento tem o mesmo tamanho, consegui comparar
    test_features = torch.tensor([truncate_pad(vocab[line], fix_lenght, vocab['<pad>']) for line in test_tokens])
    '''
    #print(train_features[0])
    #print(train_features2[0])
    #exit()
    train_iter = load_array( ( train_features, torch.tensor(y_train) ), batch_size, False)
    #for x, y in train_iter2: print(x); break
    #train_iter = d2l.load_array( ( train_features, torch.tensor(y_train) ), batch_size)
    #for x, y in train_iter: print(x); exit()
    test_iter =  load_array( ( test_features, torch.tensor(y_test) ), batch_size, False)
    return train_iter, test_iter, vocab

def init_weights(m):
    """Start weights in model"""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

def train_epoch2(net, x_train, loss, updater, y_train, batch_size):  #@save
    """The training loop defined in Chapter 3."""
    net.train() # Set the model to training mode
    loss_batch = 0; metric_batch=0; first = True
    for X, y in  x_train: # Compute gradients and update parameters
        '''print('X.shape', X.shape); exit()
        lista = []
        for x, y in train_iter: 
            if cont == 2: break
            cont+=1
            lista.append(x.numpy())
        lista = torch.Tensor(numpy.array(lista) )'''
        X = X.to(device='cuda'); y = y.to(device='cuda')

        y_hat = net(X, X, X, None)
        #print(f'len: {len(y_hat)}, y_hat: {y_hat}'); exit()
        l = loss(y_hat, y) 
        updater.zero_grad()
        l.sum().backward()
        #l.backward()
        updater.step()
        loss_batch += float( l.sum() ) * len(y) #um loss diferente para cada documento
        
        #loss_batch += (float(l) * len(y)) # calculate loss batch
        y_pred = y_hat.argmax(axis=1).tolist() # define class with max value
        y = y.cpu().tolist() # necessario ao usar GPU
        metric_batch += cv.f1([y], [y_pred])[0] * len(y) # calculate per batch metric, considerando the qtd elements
    return loss_batch/len(y_train), metric_batch/len(y_train)

def execute_model2(net, train_iter, test_iter, loss, num_epochs, updater, y_train, y_test, batch_size, mode):  #@save
    """Train a model (defined in Chapter 3)."""
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(updater, lr_lambda=lambda epoch: 0.95)
    limit_patient=5; cont_patient=0; delta=0.01 # early stop
    # base 0.01  max 0.000001
    #scheduler = torch.optim.lr_scheduler.CyclicLR(updater, base_lr=0.01, max_lr=0.1,  cycle_momentum=False)
    min_valid_loss = 10000
    list_train_loss = []; list_valid_loss = []; list_train_f1 = []; list_valid_f1 = []
    global first_epoch
    global last_epoch
    for epoch in range(num_epochs):
        if epoch+1 == num_epochs:
            last_epoch = True
        first_epoch=True
        train_loss, train_f1 = train_epoch2(net, train_iter, loss, updater, y_train, batch_size)
        if mode == 'valid':
            valid_loss, valid_f1 = evaluate_model2(net, test_iter, y_test, batch_size, loss)
            list_train_loss.append(train_loss); list_valid_loss.append(valid_loss); list_train_f1.append(train_f1); list_valid_f1.append(valid_f1)
            print(f'epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}, train_f1: {train_f1}, valid_f1: {valid_f1}, lr: {scheduler.get_last_lr()[0]}')

            scheduler.step()
            if valid_loss + delta < min_valid_loss: # early stoping, melhorar para verificar diferenca com treino
                min_valid_loss = valid_loss
                cont_patient=0
            else:
                cont_patient+=1
            if cont_patient == limit_patient: 
                print('Early Stopping')
                break
        else:
            print(f'epoch: {epoch}, train_loss: {train_loss}, train_f1: {train_f1}, lr: {scheduler.get_last_lr()[0]}')

    if mode == 'test':
        test_loss, test_f1 = evaluate_model2(net, test_iter, y_test, batch_size, loss)
        print(f'test_loss: {test_loss}, test_f1: {test_f1}')
        return test_loss, test_f1

    return list_train_loss, list_valid_loss, list_train_f1, list_valid_f1


def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, tam_vocab, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        print(f'query_size {query_size}, value_size: {value_size}')

        self.embedding = nn.Embedding(tam_vocab, 300)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        #self.classification = nn.Linear(num_hiddens, 2)
        self.classification = nn.Sequential( nn.Linear(num_hiddens, 2), nn.LogSoftmax(dim=1)  )
        #self.classification = nn.LogSoftmax(dim=1)
        #self.W_o = nn.Linear(num_hiddens, 2, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        #temp = self.embedding(queries)
        #queries = torch.transpose(queries, 0, 1)
        temp=queries
        #print(temp[0][0])
        #print('temp.shape', temp.shape)
        
        #print(torch.mean(temp[0][0],1))
        #print(temp[0].shape)
        #vetor = []
        #exit()
        #for index_doc in range( len(temp) ):
            #vetor.append(temp[index_doc].numpy() )
            #soma = 0
            #for index_token in range( len(temp[index_doc]) ):
            #    soma += temp[index_doc][index_token]
                #temp[index_doc] =  torch.sum(temp[index_doc], 1)
            #temp[index_doc] = soma

        #vetor = torch.Tensor(numpy.array(vetor) )

        #print(vetor.shape);exit()
        #print('queriessss ', temp[0][0]); exit()
        queries = temp; keys=temp; values=temp
        #queries = self.W_q(queries)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        #temp = self.W_o(output_concat)
        #print(temp); exit()
        #temp2 = torch.max(output_concat, 1)[0]
        temp2 = torch.mean(output_concat, 1)
        if last_epoch == True: # save representation
            for index_doc in range(len(temp2)):
                r_lstm.append(temp2[index_doc].cpu().detach().numpy())

        #temp2 = torch.div(torch.sum(output_concat,1), 300)
        #print(temp2);exit()
        #temp2 = torch.max(temp, 1)[0]
        temp2 = self.classification(temp2)
        #temp2 = self.soft(temp2)
        #print(temp2); exit()
        return temp2

def transpose_qkv(X, num_heads):
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

def nn_text():
    """Neural Network in pytorch"""
    ini = timeit.default_timer() # time
    index_fold=0;num_epochs = 100; mode=sys.argv[1]
    if mode != 'valid':
        index_fold=int(sys.argv[2]); dataset = sys.argv[3]; name_model = sys.argv[4] 
        dict_hyper = cv.load_dict_file(f'result_deep_learning/{dataset}_{name_model}.json')
        max_f1 = -1
        best_params = None
        for k, v in dict_hyper.items():
            if sum( dict_hyper[k]['list_valid_f1'][::-1][0:3] ) > max_f1: # sum 3 last elements
                max_f1 = sum( dict_hyper[k]['list_valid_f1'][::-1][0:3] )
                lr = dict_hyper[k]['lr']
                batch_size = dict_hyper[k]['batch_size']
                num_epochs = len(dict_hyper[k]['list_valid_f1']) # num_epoch in validation set
    else:
        batch_size = int(sys.argv[2]); lr = float(sys.argv[3]) # hyperparameter, evaluate in grid values
        dataset = sys.argv[4]; name_model = sys.argv[5]

    x_train, y_train, x_test, y_test, ids_train, ids_test, new_ids_train, new_ids_test = cv.ids_train_test_shuffle(f"dataset/{dataset}/split_5.csv", f"dataset/{dataset}/orig/texts.txt", f"dataset/{dataset}/orig/score.txt", index_fold)
    #x_train, y_train, x_test, y_test = cv.limit_data(x_train, y_train, x_test, y_test, 100)
    for index in range(len(y_train)): # cannot use negative label
        if y_train[index] == -1: y_train[index] = 0
    for index in range(len(y_test)):
        if y_test[index] == -1: y_test[index]=0
    
    x_train = [cv.preprocessor(x) for x in x_train] 
    x_test = [cv.preprocessor(x) for x in x_test]
    x_train = cv.clean_text2(x_train)
    x_test = cv.clean_text2(x_test)# tokenizar e lematizar verbo
    #x_train = cv.my_stop_word(x_train); x_test = cv.my_stop_word(x_test)
    
    if mode=='valid':
        x_train, y_train, x_test, y_test = cv.stratified_data(x_train, y_train)

    train_iter, test_iter, vocab = train_test_vocab(x_train, y_train, x_test, y_test, batch_size, 20)
    print(f'dataset: {dataset}, batch_size: {batch_size}, lr: {lr}, num_epochs: {num_epochs}')
    print(f'vocab: {len(vocab)}')
    #num_epochs=1
    if name_model == 'softmax_regression':
        net = nn.Sequential( nn.Flatten(), nn.Linear(vocab, 2) ) # neural network in n layers sequential, softmax regression
    elif name_model == 'mlp':
        net = nn.Sequential( nn.Flatten(), nn.Linear(vocab, 256), nn.ReLU(), nn.Linear(256, 2) )
    elif name_model == 'rnn':
        embed_size, num_hiddens, num_layers, devices = 300, 300, 2, d2l.try_all_gpus() 
        net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    elif name_model.__contains__('attention'):    
        num_hiddens = 768; num_heads=3 
        net = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5, len(vocab))

    net = net.to(device='cuda')
    #net.apply(init_weights)
    loss = F1_Loss().cuda()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)#, weight_decay=0.01) # otimization loss weight_decay=0.01 (evitar over)
    #trainer = torch.optim.Adam(net.parameters(), lr=lr)

    if mode == 'valid':
        list_train_loss, list_valid_loss, list_train_f1, list_valid_f1 = execute_model2(net, train_iter, test_iter, loss, num_epochs, trainer, y_train, y_test, batch_size, mode) 
    else:
        test_loss, test_f1 = execute_model2(net, train_iter, test_iter, loss, num_epochs, trainer, y_train, y_test, batch_size, mode) 
    #train_loss, train_f1, valid_f1 = execute_model(net, x_train, x_valid, loss, num_epochs, trainer, y_train, y_valid, batch_size) 
  
    if mode == 'test': # save representation to use svm
        try:
            os.mkdir("dataset/representations/"+dataset +'_' +name_model) # Create directory
        except OSError:
            print('directory exist')
        global r_lstm
        r_lstm = numpy.array(r_lstm)
        x_train = r_lstm[0:len(y_train)]; x_test = r_lstm[len(y_train):len(r_lstm)]
        x_train, y_train = cv.reorder_id(x_train, y_train, ids_train, new_ids_train)
        x_test, y_test = cv.reorder_id(x_test, y_test, ids_test, new_ids_test)
        dump_svmlight_file(x_train, y_train, f"dataset/representations/{dataset}_{name_model}/train{index_fold}")
        dump_svmlight_file(x_test, y_test, f"dataset/representations/{dataset}_{name_model}/test{index_fold}")

    file_json = f'result_deep_learning/{dataset}_{name_model}.json'
    save_dict = cv.load_dict_file(file_json)
    if mode=='valid':
        save_dict[f'batch_size: {batch_size}, lr: {lr}'] = {'lr' : lr, 'batch_size': batch_size,  'list_train_loss' : list_train_loss, 'list_valid_loss' : list_valid_loss, 'list_train_f1' : list_train_f1, 'list_valid_f1' : list_valid_f1, f'time_{mode}': timeit.default_timer() - ini}
    else:
        save_dict[f'batch_size: {batch_size}, lr: {lr}'].update({f'test_loss_fold_{index_fold}' : test_loss, f'test_f1_fold_{index_fold}' : test_f1, f'time_{mode}': timeit.default_timer() - ini })

    cv.save_dict_file(file_json, save_dict)


if __name__ == '__main__':
    nn_text()


# backup -----
#"""
#model = gensim.models.KeyedVectors.load_word2vec_format('../glove.6B.100d.txt', limit=1000)
#glove = gensim.models.KeyedVectors.load_word2vec_format('../glove.6B.300d.txt', limit=5000)

# new
'''
weights_matrix = numpy.zeros((len(vocab), 300))
for word, id_token in vocab.items():
    try:
        weights_matrix[id_token] = glove[word]
    except KeyError:
        weights_matrix[id_token] = numpy.random.normal(scale=0.6, size=(300, ))
'''

'''embeds = torch.FloatTensor(model.vectors)
embeds = []
print(vocab.keys()); exit()
for e in model.vocab:
    if e in vocab.idx_to_token:
        embeds.append(model[e])
embeds = torch.tensor(embeds)'''

#net.embedding.from_pretrained(embeds)
#weights_matrix = torch.FloatTensor(weights_matrix)
#num_embeddings, embedding_dim = weights_matrix.size()
#print(num_embeddings, embedding_dim); exit()
#"""
#net.embedding.load_state_dict({'weight': weights_matrix}) 
#net.embedding.weight.requires_grad = False
"""

    torch.save(net, f'result_deep_learning/{dataset}_{name_model}') # save model
    #torch.save(net.state_dict(), f'result_deep_learning/{dataset}_{name_model}') # save model
    #loss = nn.CrossEntropyLoss(reduction="none") # function loss, reduction="none" in example sentiment

    #x_train, y_train, x_test, y_test = cv.limit_data(x_train, y_train, x_test, y_test, 10000)

def test_model(dataset, name_model):
    index_fold=0 
    x_train, y_train, x_test, y_test = cv.ids_train_test(f"dataset/{dataset}/split_5.csv", f"dataset/{dataset}/orig/texts.txt", f"dataset/{dataset}/orig/score.txt", index_fold)
    #x_train, y_train, x_test, y_test = cv.limit_data(x_train, y_train, x_test, y_test, 1170)
    for index in range(len(y_train)): # cannot use negative label
        if y_train[index] == -1: y_train[index] = 0
    for index in range(len(y_test)):
        if y_test[index] == -1: y_test[index]=0

    x_train, y_train, x_valid, y_valid = cv.stratified_data(x_train, y_train)
    batch_size = 16
    train_iter, test_iter, vocab = train_test_vocab2(x_train, y_train, x_test, y_test, batch_size, 200)
    embed_size, num_hiddens, num_layers, devices = 300, 300, 2, d2l.try_all_gpus() 
    print(f'vocab: {len(vocab)}')
    net = torch.load(f'result_deep_learning/{dataset}_{name_model}') # load model

    loss = F1_Loss().cuda()
    test_loss, test_f1 = evaluate_model2(net, test_iter, y_test, batch_size, loss)
    print(f'test_loss: {test_loss}, test_f1: {test_f1}')

#net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
#net.load_state_dict(torch.load(f'result_deep_learning/{dataset}_{name_model}'))
#net = net.to(device='cuda')

#net.load_state_dict(torch.load("mlp.params")) # load model
#torch.save(net.state_dict(), 'mlp.params') # save model

#vectorizer = TfidfVectorizer( ) #ngram_range=(1,2)
#x_train = vectorizer.fit_transform(x_train)
#x_valid = vectorizer.transform(x_valid)
#vocab=x_train.shape[1] # the lenght entry is the vocabulary
#print(f'vocab: {vocab}')
#xx i= F.one_hot(torch.tensor([0,2]), vocab); print(xx[0]); print(xx.shape); exit()

    #for X, y in train_iter: print(f'X : {X}, y {y}')
    #exit()

    '''
    docs = []
    for doc in train_tokens: docs.append( [vocab[t]  for t in doc] )
    print(docs[0])        
    corpus = [vocab[token] for line in train_tokens for token in line]
    print(len(corpus))
    print(train_tokens[0])
    print(f'train_tokens : {len(train_tokens)}')

    exit()'''


    #corpus, vocab = corpus_vocab(x_train)
    '''
    print(len(corpus), len(vocab)); exit()
    
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    data_iter = load_array((x_train, y_train), 2)
    exit()'''
    

    #train_iter, test_iter, vocab = d2l.load_data_imdb(64)
    #print(vocab['house']); exit()
    #tokens = tokenize(x_train)
    #vocab = Vocab(tokens)

    #x_train = torch.tensor(x_train.toarray()); x_valid = torch.tensor(x_valid.toarray())
    #y_train = torch.tensor(y_train); y_valid = torch.tensor(y_valid) # convert text and label in tensor
"""


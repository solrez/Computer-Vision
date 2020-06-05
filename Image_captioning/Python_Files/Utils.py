import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import os
import torch
import nltk
from collections import Counter
import json as js
from PIL import Image
import pdb

#############################################  ENCODER  ##########################################################


class Encoder(nn.Module):
    def __init__(self, emb_dim = 512):

        super(Encoder, self).__init__()
        self.module = models.resnet34(pretrained=True)
        self.no_update_w() 

        num_feature = self.module.fc.in_features
        self.linear_layer = nn.Linear(num_feature, emb_dim)
        self.module.fc = self.linear_layer
        # pdb.set_trace()
        self.bn = nn.BatchNorm1d(emb_dim, momentum = 0.01)
        self.initialize()
        # pdb.set_trace()
    def initialize(self):
        self.linear_layer.weight.data.normal_(0.0, 0.02) 
        self.linear_layer.bias.data.fill_(0)

    def no_update_w(self):
        for param in self.module.parameters():
            param.requires_grad = False

    def forward(self, img):
        emb = self.module(img)
        return emb



#############################################  DECODER  ##########################################################
class Decoder(nn.Module):
    def __init__(self, emb_dim, h_dim, shape_voc):
        super(Decoder, self).__init__()
        self.h_dim = h_dim
        self.voc_emb = nn.Embedding(shape_voc, emb_dim)
        self.gru_mod = nn.GRU(emb_dim, h_dim)
        self.linear_layer = nn.Linear(h_dim, shape_voc)
        self.initialize()
    
    def initialize(self):
        self.voc_emb.weight.data.uniform_(-0.1, 0.1)
        self.linear_layer.weight.data.uniform_(-0.1, 0.1)
        self.linear_layer.bias.data.fill_(0)
        
    def forward(self, features, cap):
        word_len = len(cap) + 1
        emb_vocs = self.voc_emb(cap)
        emb_vocs = torch.cat((features, emb_vocs), 0)
        g_mod_output, h = self.gru_mod(emb_vocs.unsqueeze(1))
        out = self.linear_layer(g_mod_output.view(word_len, -1))
        return out

    def word_gen(self, input, h , seq_len = 20):
        inputs = input.unsqueeze(1)
        ind = []
        for i in range(seq_len):
            out, h = self.gru(inputs[i], h)
            out = self.linear(out.squeeze(1))
            _, pred = out.max(dim=1)
            ind.append(pred)
            inputs = self.word_embeddings(pred)
            inputs = inputs.unsqueeze(1)
        return ind
    
#############################################  Data Preperation  ##########################################################
class Processing():
    def __init__(self, path, words, transform):
        self.img = None
        self.collect_cap = None
        self.words = words
        self.transform = transform
        self.sentences(path)
        self.prep_img(path)
    
    def sentences(self, path):
        path_file = os.path.join(path, 'captions.txt')
        collect_cap = {}
        with open(path_file) as f:
            for row in f:
                sente = js.loads(row)
                for j, i in sente.items():
                    collect_cap[j] = i
        self.collect_cap = collect_cap
        
    def prep_img(self, path):
        path_file = os.path.join(path, 'pictures')
        files = os.listdir(path_file)
        img = {}
        for file in files:
            form_check = file.split('.')[1]
            if form_check == 'jpg':
                img[file] = self.transform(Image.open(os.path.join(path_file, file)))
        self.img = img
    
    def img_cap_prep(self):
        img = []
        cap_per = []
        for x, y in self.collect_cap.items():
            k = len(y)
            img.extend([x] * k)
            for i in y:
                cap_per.append(self.cap_per_to_Ind(i))
                
        img_cap = img, cap_per
        return img_cap
    
    def cap_per_to_Ind(self, caption):
        words = self.words
        token = nltk.tokenize.word_tokenize(caption.lower())
        token_voc = []
        token_voc.append(words.get_id('<start>'))
        token_voc.extend([words.get_id(word) for word in token])
        token_voc.append(words.get_id('<end>'))
        
        return token_voc
    
    def img_Ind(self, img_ind):
        return self.img[img_ind]      
        
def shuffle_img_cap(data, seed = 25):
    img, cap_per = data
    sh_img = []
    sh_cap_per = []
    
    k = len(img)
    torch.manual_seed(seed)
    perm_in = list(torch.randperm(k))
    for i in range(k):
        sh_img.append(img[perm_in[i]])
        sh_cap_per.append(cap_per[perm_in[i]])
        
    return sh_img, sh_cap_per


#############################################  word Preperation  ##########################################################


class voc_preperation():

    def __init__(self, cap, th):
        self.W_I = {}
        self.Ind = 0
        self.I_W = {}
        self.build_words(cap, th)
    
    def concat_voc(self, voc):
        if voc not in self.W_I:
            self.W_I[voc] = self.Ind
            self.I_W[self.Ind] = voc
            self.Ind += 1
    
    def get_id(self, voc):
        if voc in self.W_I:
            return self.W_I[voc]
        else:
            return self.W_I['<NULL>']
    
    def get_word(self, Ind):
        return self.I_W[Ind]
    
    def build_words(self, cap, th):
        count = Counter()
        tokens_voc = []
        for _, cap_per in cap.items():
            for cap in cap_per:
                cap_token = nltk.tokenize.word_tokenize(cap.lower())
                tokens_voc.extend(cap_token)
        count.update(tokens_voc)
        vocs = [voc for voc, count in count.items() if count > th]
        
        self.concat_voc('<NULL>')
        self.concat_voc('<start>')
        self.concat_voc('<end>')
        
        for voc in vocs:
            self.concat_voc(voc)
    
    def get_sentence(self, ind):
        out = ''
        for i in ind:
            voc = self.I_W[i.item()]
            out += ' ' + voc
            if voc == '<end>':
                break
        return out

def sentences(path):
    file_cap = os.path.join(path, 'captions.txt')
    collect_cap = {}
    with open(file_cap) as f:
        for x in f:
            y = js.loads(x)
            for j, i in y.items():
                collect_cap[j] = i
    return collect_cap
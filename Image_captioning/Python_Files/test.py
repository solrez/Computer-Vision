import os
import time
import pickle
import json
import torch
from PIL import Image
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb
from torchvision import transforms
from torch.autograd import Variable
from Utils import voc_preperation
from Utils import Encoder
from Utils import Decoder
import argparse



if __name__ == '__main__':
    

    argp = argparse.ArgumentParser()
    argp.add_argument("-t", "--test_dir", required=True, help="path o test image")
    argp.add_argument("-e", "--encoder", required=True, help="path to saved model")
    argp.add_argument("-d", "--decoder", required=True, help="path to saved model")
    args = vars(argp.parse_args())
    
    with open(os.path.join('../saved_model', 'txt_pi.pkl'), 'rb') as f:
        word = pickle.load(f)
    transforms = transforms.Compose([transforms.Resize((56, 56)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))])
    
    shape_voc = word.Ind
    h_dim = 512
    emb_dim = 512
  
    model_encoder =args["encoder"]
    model_decoder =args["decoder"] 
    encoder = Encoder(emb_dim)
    decoder = Decoder(emb_dim, h_dim,shape_voc)
    encoder.load_state_dict(torch.load(model_encoder, map_location='cpu'))
    decoder.load_state_dict(torch.load(model_decoder, map_location='cpu')) 
#     pdb.set_trace()
    cv2.imshow("img",cv2.imread(args["test_dir"]))
    cv2.waitKey(0)
    img = transforms(Image.open(image_path))
    img = img.unsqueeze(0)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        img = Variable(img).cuda()
    else:
        img = Variable(img)
    
    output_encoder = encoder(img)
    output_decoder = decoder.word_gen(output_encoder)

    print(word.get_sentence(output_decoder))
    
import os
import pickle as pk
import torch
import torch.nn as nn
import pdb
from torchvision import transforms
from torch.autograd import Variable
from Utils import voc_preperation
from Utils import sentences
from Utils import Processing, shuffle_img_cap
from Utils import Encoder
from Utils import Decoder
import argparse

if __name__ == '__main__':

    argp = argparse.ArgumentParser()
    argp.add_argument("-t", "--train_dir", required=True, help="path training file")
    args = vars(argp.parse_args())
    Path =args["train_dir"] 
    capi = sentences(Path)
    txt_pi = voc_preperation(capi, 5)

    with open(os.path.join('../saved_model', 'txt_pi.pkl'), 'wb') as f:
        pk.dump(txt_pi, f)

    transforms = transforms.Compose([transforms.Resize((56, 56)),transforms.ToTensor(),
                                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    process = Processing( Path,txt_pi, transforms)

    data = process.img_cap_prep()
    shape_voc = txt_pi.Ind
    lr = 1e-3
    emb_dim = 512
    h_dim = 512

    encoder = Encoder()
    decoder = Decoder( emb_dim, h_dim, shape_voc)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    print(torch.cuda.is_available())
    params = list(encoder.linear_layer.parameters()) + list(decoder.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = params, lr = lr)
    print('training... ')
    file = open('mean_loss.txt', 'a')
    for epoch in range(100):
        sh_img, sh_cap = shuffle_img_cap(data = data)
        loss_hist = []

        tic = time.time()
        for i in range(len(sh_cap)):
            image_id = sh_img[i]
            # pdb.set_trace()
            img = process.img_Ind(image_id)
            img = img.unsqueeze(0)
            # pdb.set_trace()

            if torch.cuda.is_available():
                img = Variable(img).cuda()
                cap = torch.cuda.LongTensor(sh_cap[i])
            else:
                img = Variable(img)
                cap = torch.LongTensor(sh_cap[i])

            cap_tr = cap[:-1]

            encoder.zero_grad()
            decoder.zero_grad()
            # pdb.set_trace()
            output_encoder = encoder(img)
            output_decoder = decoder(output_encoder, cap_tr)

            loss = criterion(output_decoder, cap)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss)

        mean_loss = torch.mean(torch.Tensor(loss_hist))
        file.write(str(mean_loss.item()))
        file.write(',')
        print('epoch %d mean_loss %f'%(epoch, mean_loss))
        if epoch % 5 == 0:
            torch.save(encoder.state_dict(), os.path.join('../saved_model/', 'epoch_%d_enc.pkl'%(epoch)))
            torch.save(decoder.state_dict(), os.path.join('../saved_model/', 'epoch_%d_dec.pkl'%(epoch)))
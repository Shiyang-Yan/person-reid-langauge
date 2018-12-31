from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import os
import random
from torch.autograd import Variable
import numpy as np

import gensim

VOCAB_SIZE = 6207
d_emb_dim = 512


class Language(nn.Module):
    """A CNN for text classification
    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, num_classes, dropout, batchsize):
        super(Language, self).__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_emb_dim)
        self.batchsize= batchsize
        self.hidden_dim = 512
        self.lstm = nn.LSTM(d_emb_dim, self.hidden_dim, batch_first = True)
        self.lstm_top = nn.LSTM(self.hidden_dim, self.hidden_dim,batch_first= True)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(self.hidden_dim, 256)
        self.lin_g_feature = nn.Linear(2560, 512)
    def init_hidden(self):

        if self.training:
            return (torch.zeros(1, 32, self.hidden_dim).cuda(),
                    torch.zeros(1, 32, self.hidden_dim).cuda())
        else:
            return (torch.zeros(1, self.batchsize, self.hidden_dim).cuda(),
                    torch.zeros(1, self.batchsize, self.hidden_dim).cuda())  


    def _sample_gumbel(self, shape, eps=1e-10, out=None):
  
        U1 = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
        U2 = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
        noise = - torch.log(torch.log(U2 + eps)/torch.log(U1 + eps) + eps)
        return noise

    def _gumbel_sigmoid_sample(self, logits, tau=1, eps=1e-10):
    
        dims = logits.dim()
        gumbel_noise = Variable(self._sample_gumbel(logits.size(), eps=eps, out=logits.data.new()))
        y = logits + gumbel_noise
        return F.sigmoid(y)


    def gumbel_sigmoid(self, logits, tau=0.8, hard=True, eps=1e-10):
    
        #shape = logits.size()
        #assert len(shape) == 2
        t = Variable(torch.Tensor([0.5]).cuda())
        y_soft = self._gumbel_sigmoid_sample(logits, tau=tau, eps=eps)
        if hard:
            y_hard = (y_soft>=t).float()* 1
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        return y




    def forward(self, y, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        x= x.type(torch.LongTensor).cuda()
        emb = self.emb(x)  # batch_size * 1 * seq_len * emb_dim
        #emb = x.unsqueeze(1)

        h0, c0 = self.init_hidden()
        h_top_0, c_top_0 = self.init_hidden()
        for (i,xt) in enumerate(emb.permute(1,0,2)):
              #print(xt.size())
              output, (h0, c0) = self.lstm(xt[:,None,:], (h0, c0))
              gate_features = torch.cat([h0.squeeze(), y], dim = 1)
              g_features = F.relu(self.lin_g_feature(gate_features))
              gate = self.gumbel_sigmoid(g_features)
              h_features = (h0*gate).squeeze()
              #print(h_features.size())
              output_top, (h_top_0, c_top_0) = self.lstm_top(h_features[:, None, :], (h_top_0, c_top_0))
                   
        pred = F.relu(self.lin(self.dropout(h_top_0)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

__all__ = ['ResNet50', 'ResNet101', 'ResNet50M']


d_dropout = 0.5
d_num_class = 2048


class ResNet50(nn.Module):
    def __init__(self, num_classes, batchsize, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.language = Language(d_num_class, d_dropout, batchsize)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2304, num_classes)
        #self.attention_classifier = nn.Linear(4096, 2)
        self.feat_dim = 2048

    def forward(self, x, y):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        sentence_vector = self.language(f, y)
        f_final = torch.cat([sentence_vector.squeeze(), f], dim = 1)
        alpha = self.classifier(f_final)
        #alpha_weights = F.softmax(alpha, dim=1)
        #f_a_final = torch.sum(torch.cat([sentence_vector.unsqueeze(2), f.unsqueeze(2)], dim = 2)* alpha_weights.unsqueeze(1), 2)
        if not self.training:
            return f_final
        y = self.classifier(f_final)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f_final
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=True)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.

    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072 # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        
        if not self.training:
            return combofeat
        
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


import random
import time
import numpy as np
from pandas.core import frame
from torch import nn
import torch
import torch.nn.functional as F
import math
from NETVLAD.loupe_keras import PatchNetVLAD
from AudioDistortionToken.ADT import ADTLayer
from AudioDistortionToken.QualityAnchor import QualityAnchorLayer
from AudioDistortionToken.QualityAnchorLoss import EmotionAnchorLossFunction

class TimeDistributed(nn.Module):
    def __init__(self,module,batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
        output = self.module(reshaped_input)
        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0),-1,output.size(-1))
        else:
            output = output.contiguous().view(-1,input_seq.size(1),output.size(-1))
        return output



class ADTMOS(nn.Module):
    def __init__(self):
        super(ADTMOS, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=1),nn.BatchNorm2d(16),nn.ReLU(),
                                    nn.Conv2d(16,16,(3,3),(1,1),1),nn.BatchNorm2d(16),nn.ReLU(),
                                    nn.Conv2d(16,16,(3,3),(1,3),1),nn.BatchNorm2d(16),nn.ReLU(),nn.Dropout(0.3))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=(1,1),padding=1),nn.BatchNorm2d(32),nn.ReLU(),
                                    nn.Conv2d(32,32,(3,3),(1,1),1),nn.BatchNorm2d(32),nn.ReLU(),
                                    nn.Conv2d(32,32,(3,3),(1,3),1),nn.BatchNorm2d(32),nn.ReLU(),nn.Dropout(0.3))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1),nn.BatchNorm2d(64),nn.ReLU(),
                                    nn.Conv2d(64,64,(3,3),(1,1),1),nn.BatchNorm2d(64),nn.ReLU(),
                                    nn.Conv2d(64,64,(3,3),(1,3),1),nn.BatchNorm2d(64),nn.ReLU(),nn.Dropout(0.3))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=1),nn.BatchNorm2d(128),nn.ReLU(),
                                    nn.Conv2d(128,128,(3,3),(1,1),1),nn.BatchNorm2d(128),nn.ReLU(),
                                    nn.Conv2d(128,128,(3,3),(1,3),1),nn.BatchNorm2d(128),nn.ReLU(),nn.Dropout(0.3))
        num_clusters = 3
        dim = 128
        self.NetVlad = PatchNetVLAD(num_clusters=num_clusters,dim=dim)
        self.lstm_acoustic = nn.LSTM(8,128,bidirectional=True,batch_first=True)
        self.linear_acoustic = TimeDistributed(nn.Linear(256,1),batch_first=True)
        self.ea_layer = ADTLayer(feature_size = num_clusters*dim+175+320,num_channels=6, num_audios = 1, anchor_size = 128, kernel_size = 3, stride = 1)
        self.blstm1 = nn.LSTM(896,128,bidirectional=True,batch_first=True)
        self.LN = nn.Sequential(nn.LayerNorm([399,256]),nn.ReLU())
        self.dense1 = nn.Sequential(TimeDistributed(nn.Sequential(nn.Linear(in_features=256,out_features=128),nn.ReLU()),batch_first=True)
        ,nn.Dropout(0.3)
        )
        self.frame_layer=TimeDistributed(nn.Linear(128,1),batch_first=True)
        self.average_layer = nn.AdaptiveAvgPool1d(1)
        self.linear_layer = FullyConnected(in_channels= 128, out_channels=1,activation='sigmoid',normalisation=None)
        self.listener_embedding = nn.Embedding(num_embeddings=271, embedding_dim= 256)

    def get_num_params(self):
        return sum(p.numel() for n, p in self.named_parameters())
    def forward(self,mel_input,forward_input,encoded_system,acoustic_features,listener_id,inference_mode,system_dropout,num_channels = 4):
        if inference_mode == False:
            torch.manual_seed(time.time())
            random_array = torch.rand(encoded_system.shape).cuda()
            unknown_mask = random_array < torch.tensor(system_dropout)
            encoded_system[unknown_mask] = torch.tensor(175).cuda()
        else:
            encoded_system.fill_(torch.tensor(175).cuda())
        system_embedding = torch.eye(176)[encoded_system].cuda()
        acoustic_embedding,(h_n,c_n) = self.lstm_acoustic(acoustic_features[:,:8,:].permute(0,2,1))
        acoustic_embedding = self.linear_acoustic(acoustic_embedding)
        conv1_output = self.conv1(forward_input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        x = self.NetVlad(conv4_output)[1]
        x = torch.cat([x,system_embedding,torch.squeeze(acoustic_embedding,dim=2)], dim = -1)
        ea = self.ea_layer(x.unsqueeze(1))
        conv4_output = conv4_output.permute(0,2,1,3)
        conv4_output = torch.reshape(conv4_output,(conv4_output.shape[0],conv4_output.shape[1],-1))
        blstm_output,(h_n,c_n) = self.blstm1(conv4_output)
        ln_output = self.LN(blstm_output)
        fc_output = self.dense1(ln_output)
        frame_score = self.frame_layer(fc_output)
        uttr_score = self.average_layer(frame_score.permute(0,2,1))
        mos_score = (uttr_score + 0.1*self.linear_layer(ea.unsqueeze(2).permute(0,2,1))).squeeze(1)
        return torch.reshape(uttr_score,(uttr_score.shape[0],-1)),frame_score,mos_score
    
    def cal_mean_score(self,forward_input):
        conv1_output = self.conv1(forward_input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv4_output = conv4_output.permute(0,2,1,3)
        conv4_output = torch.reshape(conv4_output,(conv4_output.shape[0],conv4_output.shape[1],-1))
        blstm_output,(h_n,c_n) = self.blstm1(conv4_output)
        ln_output = self.LN(blstm_output)
        fc_output = self.dense1(ln_output)
        frame_score = self.frame_layer(fc_output)
        avg_score = self.average_layer(frame_score.permute(0,2,1))
        return torch.reshape(avg_score,(avg_score.shape[0],-1))
    
    
class ADTLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(ADTLoss, self).__init__()
        self.sigma = sigma
        self.fn_huber_loss1 = nn.SmoothL1Loss()
        self.fn_huber_loss2 = nn.SmoothL1Loss()
        self.fn_huber_loss3 = nn.SmoothL1Loss()
        self.quality_anchor_loss = EmotionAnchorLossFunction(alpha= 1)

    def forward(self, avg_score_true, frame_score_true,avg_score,frame_score,uttr_score):
        bias_score = avg_score - uttr_score
        frame_score_true = frame_score_true.half() - bias_score.expand(-1,frame_score_true.shape[1]).unsqueeze(-1).half()
        contrastive_loss = self.contrastiveLoss(avg_score,avg_score_true)
        return self.fn_huber_loss1(avg_score_true,avg_score) + self.fn_huber_loss2(frame_score_true,frame_score), contrastive_loss


class FullyConnected(nn.Module):
    """
    Creates an instance of a fully-connected layer. This includes the
    hidden layers but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, activation, normalisation,
                 att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == 'global':
            self.act = None
            self.norm = None
        elif activation == 'tanh':
            self.act = nn.Tanh()
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == 'bn':
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == 'wn':
                self.wnf = nn.utils.weight_norm(self.fc, name='weight')

        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.fc)
        else:
            init_layer(self.fc)
        if self.norm == 'bn':
            init_bn(self.bnf)

    def forward(self, input):
        """
        Passes the input through the fully-connected layer
        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm is not None:
            if self.norm == 'bn':
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)
            else:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)

        return x


def init_att_layer(layer):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.
    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    layer.weight.data.fill_(1.)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.
    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    Input
        layer: torch.Tensor - The current layer of the neural network
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

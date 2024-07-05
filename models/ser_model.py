"""
AIO -- All Model in One
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model
from transformers import HubertModel

from models.ser_spec import SER_AlexNet
from models.newnet import MF


class Ser_Model2(nn.Module):
    def __init__(self):
        super(Ser_Model2, self).__init__()
        
        # CNN for Spectrogram
        # self.alexnet_model = SER_AlexNet(num_classes=4, in_ch=3, pretrained=True)

        self.newnet = MF()
        
        self.post_spec_dropout = nn.Dropout(p=0.1)
        # self.post_spec_layer = nn.Linear(9216, 128) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        self.post_spec_layer = nn.Linear(288, 128) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l

        self.post_spec_layer2 = nn.Linear(288, 149)  # fusion hubert
        
        # LSTM for MFCC        
        # self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True) # bidirectional = True

        # self.post_mfcc_dropout = nn.Dropout(p=0.1)
        # self.post_mfcc_layer = nn.Linear(153600, 128) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm
        
        # Spectrogram + MFCC  
        self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        self.post_spec_mfcc_att_layer = nn.Linear(256, 149) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
                        
        # WAV2VEC 2.0
        # self.wav2vec2_model = Wav2Vec2Model.from_pretrained("/home/jxx/code/CA-MSER-main-usually/wav2vec2-base-960h")
        self.wav2vec2_model = HubertModel.from_pretrained("/home/jxx/pretrained/hubert-base-ls960")
        # self.wav2vec2_model = HubertModel.from_pretrained("/home/jxx/pretrained/hubert-large-ls960-ft")


        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(768, 128) # 512 for 1 and 768 for 2
        
        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(256, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)

        self.avgpool = nn.AdaptiveAvgPool2d((12, 64))

        # MCA
        self.lstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)
        self.bertpool = nn.AdaptiveAvgPool2d((32, 24))
        self.mca_layer = nn.Linear(1024, 768)
        
                                                                     
    def forward(self, audio_spec, audio_mfcc, audio_wav):      
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]

        # MF for spec
        audio_spec = self.newnet(audio_spec)
        
        audio_spec_d = self.post_spec_dropout(audio_spec) # [batch, 9216]  
        audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=False) # [batch, 128]  

        audio_spec_rh = F.relu(self.post_spec_layer2(audio_spec_d), inplace=False)
        audio_spec_rh = audio_spec_rh.reshape(audio_spec_rh.shape[0], 1, -1)  # [batch, 1, 149]
        

        # wav2vec 2.0 
        audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state # [batch, 149, 768] 
        
        # MCA用 BiLSTM输出的特征
        bert2, _ = self.lstm(audio_wav)

        audio_wav2 = torch.matmul(audio_spec_rh, bert2) # [batch, 1, 768]

        audio_wav = self.bertpool(audio_wav) # [batch, 32, 24]

        audio_wav2 = audio_wav2.reshape(audio_wav2.shape[0], 24, 32)

        audio_wav = torch.matmul(audio_wav, audio_wav2)
        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) #[batch, 1024]
        audio_wav = self.mca_layer(audio_wav)   # [batch, 768]     
        


        # audio_wav = torch.matmul(audio_spec_rh, audio_wav) # [batch, 1, 768] # 修改为不用共同注意 因此注释掉这一行
        # audio_wav = self.avgpool(audio_wav)   # [batch,12,64] 无共同注意用

        # audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) # [batch, 768] 
        
        audio_wav_d = self.post_wav_dropout(audio_wav) # [batch, 768]

        audio_wav_p = F.relu(self.post_wav_layer(audio_wav_d), inplace=False) # [batch, 128] 
        
        ## combine()
        audio_att = torch.cat([audio_spec_p, audio_wav_p], dim=-1)  # [batch, 256] 
        audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 256] 
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False) # [batch, 128] 
        audio_att_d_2 = self.post_att_dropout(audio_att_1) # [batch, 128] 
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128] 
        output_att = self.post_att_layer_3(audio_att_2) # [batch, 4] 
        
  
        output = {
            'F1': audio_wav_p,
            'F2': audio_att_1,
            'F3': audio_att_2,
            'F4': output_att,
            'M': output_att
        }            
        

        return output

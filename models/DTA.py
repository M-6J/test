from models.TXA import T_XA
from models.TNA import T_NA
import torch.nn as nn

class DTA(nn.Module):
    def __init__(self, T, out_channels):
        super().__init__()

        self.T_NA = T_NA(in_planes=out_channels*T, kernel_size=7)
        self.T_XA = T_XA(time_step=T) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_seq, spikes):
        
        B, T, C, H, W = x_seq.shape 
        
        x_seq_2 = x_seq.reshape(B, T*C, H, W)
        T_NA = self.T_NA(x_seq_2) 
        T_NA = T_NA.reshape(B, T, C, H, W)
        
        T_XA = self.T_XA(x_seq) 
        
        out = self.sigmoid(T_NA * T_XA)
        y_seq = out * spikes  

        return y_seq   
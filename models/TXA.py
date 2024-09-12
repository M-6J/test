import torch.nn as nn
import torch 

class T_XA(nn.Module):
    def __init__(self, time_step):
        super(T_XA, self).__init__()
        self.conv_t = nn.Conv1d(in_channels=time_step, out_channels=time_step, 
                              kernel_size=3, padding='same', bias=False)
        
        self.conv_c = nn.Conv1d(in_channels=64, out_channels=64,
                                kernel_size=6, padding='same', bias=False)

        self.sigmoid = nn.Sigmoid()
        
        self.scale_t = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.scale_c = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        
    def forward(self, x_seq):    
        x_t = torch.mean(x_seq.permute(0, 1, 2, 3, 4), dim=[3, 4]) 
        x_c = x_t.permute(0, 2, 1) 

        conv_t_out = self.conv_t(x_t)
        attn_map_t = self.sigmoid(conv_t_out)

        conv_c_out = self.conv_c(x_c).permute(0, 2, 1)  
        attn_map_c = self.sigmoid(conv_c_out)
        
        after_scale_t = attn_map_t * self.scale_t 
        after_scale_c = attn_map_c*self.scale_c
        
        attn_t_ft = x_seq + after_scale_t[:, :, :, None, None]
        attn_c_ft = x_seq + after_scale_c[:, :, :, None, None]

        y_seq = attn_t_ft * attn_c_ft 

        return y_seq

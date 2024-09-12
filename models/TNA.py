import torch.nn as nn


class T_NA(nn.Module): 
    def __init__(self, in_planes, kernel_size=21, attn_shortcut=True):
        super().__init__()
        
        self.encoding = nn.Sequential(nn.Conv2d(in_planes, in_planes, 1), nn.GELU() )
        self.tca = TCA(in_planes, kernel_size)
        self.decoding = nn.Conv2d(in_planes, in_planes, 1)      
        self.attn_shortcut = attn_shortcut

    def forward(self, x): 

        if self.attn_shortcut:
            shortcut = x.clone()

        x = self.encoding(x)
        x = self.tca(x)
        x = self.decoding(x)
        
        if self.attn_shortcut:
            x = x + shortcut
        return x


class TCA(nn.Module):
    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1 
        d_p = (d_k - 1) // 2 
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1) 
        dd_p = (dilation * (dd_k - 1) // 2) 
        self.LTCA = nn.Sequential(
            nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim),
            nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation),
            nn.Conv2d(dim, dim, 1)
        )
        self.reduction = max(dim // reduction, 4) 
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.MLP_block = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), 
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False))


    def forward(self, x):
        u = x.clone()     
        LTCA_attn = self.LTCA(x)      

        b, c, _, _ = x.size()
        GAP_attn = self.GAP(x).view(b, c)    
        GTCA_attn = self.MLP_block(GAP_attn).view(b, c, 1, 1)
        
        return GTCA_attn * LTCA_attn * u 

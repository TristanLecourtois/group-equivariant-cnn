import torch
import torch.nn.init as init
import math


class Conv2dZ2P4(torch.nn.Module):
    # convolution de Z2 vers P4 donc d'une image classique à une feature map structuré ( avec 4 orientations )
    def __init__(self, in_channels, out_channels, kernel_size, g_type="p4",  
                 dilation=1, groups=1, bias=False, device="cuda", dtype=None, *args, **kwargs):
        super().__init__()
        
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size) # classique pour une conv normale
        self.weight = torch.nn.Parameter(w).to(device)
        
        self.g_type = g_type # g_type peut-être soit p4 soit p4m
        # à partir des poids weights on l'étend au group convolution
        self.get_kernel = get_p8weight if self.g_type == "p4m" else get_p4weight 

        self.gconv_dim = 8 if self.g_type == "p4m" else 4 # pour p4m = 4 rotations + 4 flips 
        self.__args = args
        self.__kwargs = kwargs
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        self.__weight_initialization()
    
    def forward(self, x):
        # input x : [B,Cin, H,W] B = batch nb image 
        w = self.get_kernel(self.weight) # (K * g_dim, C_in, k, k)
        
        y = torch.nn.functional.conv2d(x, w, *self.__args, **self.__kwargs)# (B, g_dim * K, H', W')
        y = y.view(y.size(0), -1, 4, y.size(2), y.size(3)) # (B, K', 4, H', W')
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y
            
    def __weight_initialization(self):
        # l'init de Kaiming uniform
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # préserver la variance des activations entre couches (éviter disparition/explosion du signal).
        if self.bias is not None: 
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


def get_p4weight(w):
    # input: [C, K, H, W]
    # output: [4*C, K, H, W]
    
    # on génère les 4 versions transformées (rotated) du filtre de base
    ws = [torch.rot90(w, k, (-2, -1)) for k in range(4)]
    return torch.cat(ws, 1).view(-1, w.size(1), w.size(2), w.size(3))

def get_p8weight(w):
    #même chose mais 8 versions pour p4m 
    # input: [K, C, H, W]
    # output: [8*K, C, H, W]
    w_p4 = get_p4weight(w)
    return torch.cat([w_p4, torch.flip(w_p4, dims=(-1,))], 1).view(-1, w.size(1), w.size(2), w.size(3))


def g_rot4(x, k, reverse=False):
    # rotation spatial : on tourne la feature map dans le plan H×W
    # rotation des orientations : on décale les 4 canaux correspondant aux orientations pour que chaque orientation corresponde à la nouvelle rotation
    device = x.device
    if reverse: 
        k = -k
    x = torch.rot90(x, k, (-2, -1))
    return torch.roll(x, k, dims=-3).to(device)


class Conv2dP4P4(torch.nn.Module):
    # convolution de P4 vers P4 donc d'une feature map strucutrée à une feature map structurée
    def __init__(self, in_channels, out_channels, kernel_size, g_type="p4", bias=False, device="cuda", *args, **kwargs):
        # in_channels : nb_canaux ( feature map avec orientation)
        super().__init__()
        self.out_channels = out_channels
        w = torch.empty(out_channels*4, in_channels, kernel_size, kernel_size)# chaque filtre est appliqué aux 4 orientations de l’entrée
        self.weight = torch.nn.Parameter(w).to(device)
        
        self.g_type = g_type 
        self.get_kernel = get_p8weight if self.g_type == "p4m" else get_p4weight
        self.gconv_dim = 8 if self.g_type == "p4m" else 4
        self.__args = args
        self.__kwargs = kwargs
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        self.__weight_initialization()
    
    def forward(self, x):
        # x : feature map structuré [B, C_in, 4, H, W]
        b = self.bias.repeat(self.gconv_dim) if self.bias is not None else None
        w = self.weight
        B, C, _, H, W = x.shape
        y = None
        
        device = x.device
        
        for i in range(4):
            _, _, _, H, W = x.shape
            x_ = g_rot4(x, -i) # fait tourner l’entrée pour correspondre à l’orientation du filtre.

            x_ = x_.transpose(1,2).reshape(B, C * self.gconv_dim, H, W) # [B, C_in * 4, H, W] reshape pour appliquer la convolution 2D classique

            t = torch.nn.functional.conv2d(x_, w, groups=self.gconv_dim, *self.__args, **self.__kwargs) # applique chaque sous-groupe de filtres à chaque orientation séparément.
            _, _, H, W = t.shape
            t = t.reshape(B, -1, 4, H, W).sum(dim=2) # applique chaque sous-groupe de filtres à chaque orientation séparément.

            if y is None: 
                y = torch.zeros(B, self.out_channels, 4, H, W).to(device)
            y[:, :, i, :, :] = t # on écrit la sortie correspondant à l’i-ème orientation du groupe
            
        if self.bias is not None:
            y = y + b
        return y
            
    def __weight_initialization(self):
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None: 
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound) 



   
class MaxPoolingP4(torch.nn.Module):
    def __init__(self, kernel_size=(2,2), stride=2):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size)
        
    def forward(self, x):
        B, C, _, H, W = x.shape
        x = x.view(B, -1, H, W)  # [B, C*4, H, W] 
        # on fait ça car torch.nn.MaxPool2d ne connait pas la dimension des orientations : Les orientations sont traitées indépendamment, exactement comme si elles étaient des canaux séparés.
        x_pool = self.pool(x)
        _, _, H, W = x_pool.shape
        return x_pool.view(B, C, 4, H, W) #[B, C, 4, H, W]
    

class AvgPoolingP4(torch.nn.Module):
    def __init__(self, kernel_size=(2,2), stride=2):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size)
        
    def forward(self, x):
        B, C, _, H, W = x.shape
        x = x.view(B, -1, H, W)  # [B, C*4, H, W] 
        x_pool = self.pool(x)
        _, _, H, W = x_pool.shape
        return x_pool.view(B, C, 4, H, W)#[B, C, 4, H, W]


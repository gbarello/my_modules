import torch
import torch.nn as nn
import numpy as np

class stop_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        return x
    @staticmethod
    def backward(ctx,x):        
        return torch.zeros_like(x)

    
stop_gradients = stop_grad.apply

def safe_torch_expnorm(W,dim):
    MAX = torch.max(W,dim = dim,keepdims = True)[0]
    DIFF = W - dim
    return torch.exp(DIFF)/torch.sum(torch.exp(DIFF),dim = dim,keepdims = True)

class my_ResLayer(nn.Module):
    def __init__(self,out_size,bottleneck_size,kernel_size,nonlinearity = torch.relu):
        super(my_ResLayer,self).__init__()
        
        self.out_size = out_size
        self.bottleneck_size = bottleneck_size
        self.nonlinearity = nonlinearity
        self.padding = kernel_size // 2
        
        
        self.L1 = nn.Conv2d(self.out_size,self.bottleneck_size,kernel_size, padding = self.padding)
        self.L2 = nn.Conv2d(self.bottleneck_size,self.out_size,1)
        
    def forward(self,x):
        y = self.L1(x)
        y = self.nonlinearity(y)
        y = self.L2(y)
        
        return x + y
    
class my_ResBlock(nn.Module):
    def __init__(self,n_layers,out_size,bottleneck_size,kernel_size,nonlinearity = torch.relu):
        super(my_ResBlock,self).__init__()
        
        self.layers = nn.ModuleList([my_ResLayer(out_size,bottleneck_size,kernel_size,nonlinearity) for k in range(n_layers)])
        self.batchnorm = nn.ModuleList([nn.BatchNorm2d(out_size) for k in range(n_layers)])
        
    def forward(self,x):
        y = x
        for L,B in zip(self.layers,self.batchnorm):
            #y = B(L(y))
            y = L(y)
        return y
        
class ffw(nn.Module):
    def __init__(self,layers,nonlinearity = lambda x:x):
        super(ffw,self).__init__()
        self.layers = nn.ModuleList(layers)
        self.nonlinearity = nonlinearity
    def forward(self,x):
        y = x
        for L in self.layers:
            y = self.nonlinearity(L(y))
        return y
    
class my_ResNet(nn.Module):
    def __init__(self,n_layers,in_size,bottleneck_size,kernel_size,stride = 1,nonlinearity = torch.relu, mean = True):
        super(my_ResNet,self).__init__()
        
        if isinstance(stride,int):
            stride = [stride for k in kernel_size]
        
        self.blocks = []
        
        for i,(n,b,k) in enumerate(zip(n_layers,bottleneck_size,kernel_size)):
            self.blocks.append(ffw([
                nn.Conv2d(in_size[i],in_size[i+1],stride[i]),
                nn.BatchNorm2d(in_size[i+1],track_running_stats = True),
                my_ResBlock(n,in_size[i+1],b,k,nonlinearity = nonlinearity)])
                              )
            
        self.blocks = nn.ModuleList(self.blocks)
        self.mean = mean
        self.pool = nn.MaxPool2d(2,stride = 2)
    def forward(self,x,debug = False):
        y = x
        for i,L in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                y = self.pool(L(y))
            else:
                y = L(y)
        if self.mean:
            return torch.mean(y,(-2,-1))
        else:
            return y
class InfoNCE_loss(nn.Module):
    def __init__(self,context_size,encoding_size,fixed = False):
        super(InfoNCE_loss, self).__init__()
        
        self.fixed = fixed
        if self.fixed:
            assert encoding_size == context_size
            self.L1 = lambda x:x
        else:
            self.L1 = nn.Linear(encoding_size,context_size,bias = False)
            
    def transform(self,x,y):
        return torch.sum(x*self.L1(y),-1)
    
    def generate_NCE(self,both):
        expmax = torch.max(both,-1,keepdims = True)[0]
        expsum = both - expmax

        tops = torch.squeeze(torch.index_select(both,-1,torch.tensor(0).to(both.device)),-1)
        extra = torch.log(torch.sum(torch.exp(expsum),-1)) + torch.sum(expmax,-1)

        InfoNCE = tops - extra
        
        return torch.mean(InfoNCE)
        
    def forward(self,C,x1,x2):
        #C is the context vectors
        #x1 is a batch of videos [nbatch, n_features]
        #x2 is a batch of negative samples [nbatch, nsample, n_features]
        positive = self.transform(C,x1)
        negative = self.transform(torch.unsqueeze(C,-2),x2)
        both = torch.cat([torch.unsqueeze(positive,-1),negative],-1)
        
        return self.generate_NCE(both)

    
class InfoNCE_time_loss(nn.Module):
    def __init__(self,context_size,encoding_size):
        super(InfoNCE_time_loss, self).__init__()
        self.InfoNCE = InfoNCE_loss(context_size,encoding_size)
        
    def forward(self,C,x1,x2):
        #C: [batch,time,feature]
        #x1: [batch,time,feature]
        #x2: [batch,time,sample,feature]
        out = self.InfoNCE(torch.flatten(C,0,1),torch.flatten(x1,0,1),torch.flatten(x2,0,1))
        return out
    
class Multi_InfoNCE_time_loss(nn.Module):
    def __init__(self,c_size,x_size,dt):
        super(Multi_InfoNCE_time_loss,self).__init__()
        self.dt = dt
        self.InfoNCE = nn.ModuleList([InfoNCE_time_loss(c_size,x_size) for t in self.dt])

    def forward(self,c,x1,x2):
        #[batch,time,feature]
        #x2 shape : [batch,time, sample, feature]
        
        out = [self.InfoNCE[i](c[:,:-t],x1[:,t:],x2[:,t:]) for i,t in enumerate(self.dt)]
        return torch.mean(torch.stack(out))
        
class MINE_loss(nn.Module):
    def __init__(self,context_size,encoding_size):
        super(MINE_loss, self).__init__()
        
        self.L1 = nn.Linear(encoding_size,context_size,bias = False)
        
    def transform(self,x,y):
        return torch.sum(y*self.L1(x),-1)
    
    def forward(self,C,x1,x2):
        #C is the context vectors
        #x1 is a batch of videos [nbatch, ntime, n_features]
        #x2 is a batch of negative samples [nbatch, ntime, nsample, n_features]
        
        positive = self.transform(C,x1)#torch.sum(C*self.transform(x1),dim = -1)
        negative = self.transform(torch.unsqueeze(C,1),x2)#torch.sum(torch.unsqueeze(C,2)*self.transform(x2),dim = -1)

        expmax = torch.max(negative)
        expsum = negative - expmax
        
        MINE = torch.mean(positive) - torch.log(torch.mean(torch.exp(expsum))) - expmax
        
        return MINE
    

class MINE_time_loss(nn.Module):
    def __init__(self,context_size,encoding_size):
        super(MINE_time_loss, self).__init__()
        self.MINE = MINE_loss(context_size,encoding_size)
        
    def forward(self,C,x1,x2):
        #C: [batch,time,feature]
        #x1: [batch,time,feature]
        #x2: [batch,time,sample,feature]
        
        return self.MINE(torch.flatten(C,0,1),torch.flatten(x1,0,1),torch.flatten(x2,0,1))
    
def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]
    
class set_net(nn.Module):
    def __init__(self,n_input, layers):
        super(set_net,self).__init__()
        self.sizes = [n_input] + list(layers)
        self.layers = nn.ModuleList([nn.Linear(2*l,self.sizes[i+1]) for i,l in enumerate(self.sizes[:-1])])
            
    def apply(self,x):
        #x is expected to have shape [n_item, n_input]
        y = x
        for i,l in enumerate(self.layers):
            mean = torch.mean(y,0).view([1,y.shape[1]])
            y = torch.tanh(l(torch.cat([y,mean.expand(y.shape)],1)))
            
        return torch.mean(y,0)
    
    def forward(self,x):
        return torch.stack([self.apply(i) for i in x],0)
    
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def causal_conv_1d_(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    pad = (kernel_size - 1) * dilation
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

class CausalConv1D(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1D,self).__init__()
        self.conv1 = causal_conv_1d_(in_channels, out_channels, kernel_size, dilation=1, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]//self.conv1.stride[0]]  # remove trailing padding
        return x
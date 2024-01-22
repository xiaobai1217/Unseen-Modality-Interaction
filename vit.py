import torch
from torch import nn
import pdb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

def MySoftmax(x, dim=-1, mask=None):
    maxes = torch.max(x*mask, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)

    if mask is not None:
        x_exp_sum = torch.sum(x_exp*mask, dim, keepdim=True)
        return x_exp * mask / (x_exp_sum + 1e-10)
    else:
        x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
        return x_exp / x_exp_sum

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.fn = fn
    def forward(self, x):
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, branch_num):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        qs = qkv[0]
        ks = qkv[1]
        vs = qkv[2]
        per_branch = int((qs.size()[1] - branch_num) // branch_num)
        outputs = []
        cls_outputs = []
        for i in range(branch_num):
            q = torch.cat((qs[:, i:(i+1), :], qs[:, (i*per_branch+branch_num) : ((i+1)*per_branch + branch_num), :] ), dim=1)
            q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
            k = torch.cat((ks[:, i:(i+1), :], ks[:, (i*per_branch+branch_num) : ((i+1)*per_branch + branch_num), :] ), dim=1)
            k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
            v = torch.cat((vs[:, i:(i+1), :], vs[:, (i*per_branch+branch_num) : ((i+1)*per_branch + branch_num), :] ), dim=1)
            v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            cls_outputs.append(out[:,:1])
            outputs.append(out[:,1:])
        out = torch.cat(outputs, dim=1)
        cls_outputs = torch.cat(cls_outputs, dim=1)
        out = torch.cat((cls_outputs, out), dim=1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn = self.attend(dots)

        # device = x.get_device()
        # branch_masks = branch_masks.to(device)
        # attn_branches = 0
        # for i in range(branch_masks.size()[1]):
        #     attn = MySoftmax(dots, dim = -1, mask = branch_masks[:,i])#self.attend(dots)
        #     attn_branches += attn
        # attn = self.dropout(attn)

        # out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, branch_num):
        for pn1, attn, pn2, ff in self.layers:
            x1 = pn1(x)
            x = attn(x1, branch_num) + x
            x1 = pn2(x)
            x = ff(x1) + x
        return x

class ViT(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., num_position = 512, branch_num = 2):
        super().__init__()
        self.mask_ratio = 0.5

        self.audio_embedding = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, dim),
            nn.LayerNorm(dim)
        )
        self.rgb_embedding = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, dim),
            nn.LayerNorm(dim)
        )

        per_branch = num_position // branch_num
        branch_masks = []
        for i in range(branch_num):
            branch_mask1 = torch.zeros((1, 1, heads, num_position + branch_num, num_position + branch_num))
            branch_mask1[:, :, :, (i*per_branch + branch_num) :((i+1)*per_branch + branch_num), (i*per_branch + branch_num) :((i+1)*per_branch + branch_num)] = 1
            branch_mask1[:, :, :, i, (i*per_branch + branch_num) :((i+1)*per_branch + branch_num)] = 1 # cls mask
            branch_mask1[:, :, :, (i*per_branch + branch_num) :((i+1)*per_branch + branch_num), i] = 1 # cls mask
            branch_mask1[:,:,:,i,i] = 1
            branch_masks.append(branch_mask1)
        self.branch_masks = torch.cat(branch_masks, dim=1) #[1, branch_num, heads, num_position, num_position]
        self.branch_num = branch_num

        self.pos_embedding = nn.Parameter(torch.randn(1, num_position + branch_num, dim))
        self.cls_token = nn.Parameter(torch.randn(1, branch_num, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, rgb, audio, rgb_mask, audio_mask):
        rgb = self.rgb_embedding(rgb)
        audio = self.audio_embedding(audio)
        
        features = rgb * rgb_mask + audio * audio_mask
        
        b, n, _ = features.shape
        x = features + self.pos_embedding[:, self.branch_num:]

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        cls_tokens = cls_tokens + self.pos_embedding[:, :self.branch_num]
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.transformer(x, self.branch_num)

        x = x[:, :self.branch_num]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x
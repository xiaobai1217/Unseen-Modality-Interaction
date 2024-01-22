import torch
from torch import nn
import pdb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTReorganization(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., num_position = 512):
        super().__init__()

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

        self.embedding = {
            'Audio': self.audio_embedding,
            'RGB': self.rgb_embedding,
        }

        self.rgb_pos_embedding = nn.Parameter(torch.randn(1, 16*7*7, dim))
        self.audio_pos_embedding = nn.Parameter(torch.randn(1, 146, dim))

        self.pos_embedding = {
            'Audio': self.audio_pos_embedding,
            'RGB': self.rgb_pos_embedding,
        }

        self.dropout = nn.Dropout(emb_dropout)

        self.rgb_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.audio_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.transformer = {
            'RGB': self.rgb_transformer,
            'Audio': self.audio_transformer,
        }

        self.pool = pool
        self.to_latent = nn.Identity()

        self.rgb_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_position)
        )

        self.audio_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_position)
        )

        self.heads = {
            'RGB': self.rgb_head,
            'Audio': self.audio_head,
        }
    
    def reorganization(self, feature, position):
        feature = torch.matmul(
            feature.transpose(1, 2).contiguous(), position
        )
        feature = feature.transpose(1, 2).contiguous()
        return feature

    def forward(self, rgb_inputs, audio_inputs):

        rgb = self.rgb_embedding(rgb_inputs) + self.rgb_pos_embedding
        audio = self.audio_embedding(audio_inputs) + self.audio_pos_embedding

        rgb = self.dropout(rgb)
        audio = self.dropout(audio)

        rgb = self.rgb_transformer(rgb)
        audio = self.audio_transformer(audio)

        rgb = self.rgb_head(rgb)
        audio = self.audio_head(audio)

        rgb = torch.softmax(rgb, dim = -1)
        audio = torch.softmax(audio, dim=-1)

        rgb = self.reorganization(rgb_inputs, rgb)
        audio = self.reorganization(audio_inputs, audio)

        return rgb, audio
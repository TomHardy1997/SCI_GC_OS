import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.utils.checkpoint import checkpoint

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim=128, hidden_dim=256, dropout=0.1):
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
    def __init__(self, dim=128, heads=2, dim_head=32, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x, register_hook=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn_norm(x)
        x = checkpoint(self.attn, x, use_reentrant=False)
        x = self.ff_norm(x)
        x = checkpoint(self.ff, x, use_reentrant=False)
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        input_dim=512,
        dim=128,
        depth=1,
        heads=2,
        mlp_dim=128,
        pool='cls',
        dim_head=32,
        dropout=0.1,
        emb_dropout=0.1,
        pos_enc=None,
    ):
        super(Transformer, self).__init__()
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (class token) or mean pooling'

        self.projection = nn.Sequential(nn.Linear(input_dim, dim, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.transformer = nn.ModuleList([TransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.pos_enc = pos_enc

    def forward(self, x, coords=None, register_hook=False):
        b, n, _ = x.shape
        # print("Original shape of x:", x.shape)
        x = self.projection(x)

        if self.pos_enc:
            x = x + self.pos_enc(coords)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        for layer in self.transformer:
            x = layer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # import ipdb;ipdb.set_trace()
        return self.mlp_head(self.norm(x))

# 测试模型定义是否正确
if __name__ == "__main__":
    transformer = Transformer(num_classes=4)
    output = transformer(torch.rand(1, 1000, 512))  # 输入长度为 1000，维度为 512
    print(output)

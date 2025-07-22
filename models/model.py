import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# pip install native-sparse-attention-pytorch
from native_sparse_attention_pytorch import SparseAttention

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

# For Standard ViT
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
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

# For SparseViT
class SparseTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., **sparse_attn_kwargs):
        super().__init__()
        assert SparseAttention is not None, 'native-sparse-attention-pytorch is not installed'
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SparseAttention(dim, heads = heads, dim_head = dim_head, **sparse_attn_kwargs)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# Standard ViT Model
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

# Sparse ViT Model
class SparseViT(nn.Module):
    def __init__(
        self, *, image_size, num_classes, dim, depth, heads, mlp_dim, 
        pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
        # SparseAttention args
        sliding_window_size = 32,
        compress_block_size = 16,
        compress_block_sliding_stride = 4,
        selection_block_size = 64,
        num_selected_blocks = 4
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(1) # Using 1x1 patches (pixels as tokens)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        sparse_attn_kwargs = dict(
            sliding_window_size = sliding_window_size,
            compress_block_size = compress_block_size,
            compress_block_sliding_stride = compress_block_sliding_stride,
            selection_block_size = selection_block_size,
            num_selected_blocks = num_selected_blocks,
            causal = False # Not an autoregressive model
        )

        self.transformer = SparseTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, **sparse_attn_kwargs)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# ViT-B/16을 기반으로 한 일반 ViT 모델 정의
def vit_b16(num_classes, image_size):
    return ViT(
        image_size=image_size,
        patch_size=16,  # 표준 ViT-B/16의 패치 크기
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        channels=3
    )

# ViT-B/16을 기반으로 한 SparseViT 모델 정의
def sparse_vit_b16(num_classes, image_size):
    return SparseViT(
        image_size=image_size,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        channels=3,
        # SparseAttention 기본 하이퍼파라미터 설정
        sliding_window_size = 32,
        compress_block_size = 16,
        compress_block_sliding_stride = 4,
        selection_block_size = 64,
        num_selected_blocks = (image_size // 64)**2
    )

# 더 공격적인 희소성 설정으로 VRAM 사용량을 줄인 버전
def sparse_vit_b16_efficient(num_classes, image_size):
    return SparseViT(
        image_size=image_size,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        channels=3,
        # VRAM 사용량을 줄이기 위해 파라미터 조정
        sliding_window_size = 16,            # 지역 어텐션 범위 감소 (32 -> 16)
        compress_block_size = 16,
        compress_block_sliding_stride = 8,   # 요약본 생성을 더 거치게 함 (4 -> 8)
        selection_block_size = 64,
        num_selected_blocks = 1              # 전역 어텐션 블록 수를 1개로 고정
    )

class Temp(nn.Module):
    def __init__(self, args):
        super().__init__()
        # args에 따라 모델을 동적으로 선택
        # 예시: python main.py --model vit ...
        #       python main.py --model sparse_vit ...
        
        # config.py에 num_classes, image_size 인자 추가 필요
        num_classes = getattr(args, 'num_classes', 10) # default 10
        image_size = getattr(args, 'image_size', 224) # default 224

        if args.model == 'vit_b16':
            self.model = vit_b16(num_classes=num_classes, image_size=image_size)
        elif args.model == 'sparse_vit_b16':
            if SparseAttention is None:
                raise ImportError("Please install native-sparse-attention-pytorch to use SparseViT")
            self.model = sparse_vit_b16(num_classes=num_classes, image_size=image_size)
        elif args.model == 'sparse_vit_b16_efficient':
            if SparseAttention is None:
                raise ImportError("Please install native-sparse-attention-pytorch to use SparseViT")
            self.model = sparse_vit_b16_efficient(num_classes=num_classes, image_size=image_size)
        else:
            raise ValueError(f"Unknown model type: {args.model}")

    def forward(self, x):
        return self.model(x)
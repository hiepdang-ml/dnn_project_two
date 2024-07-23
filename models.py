from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchPositionEmbedding(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        patch_size: int, 
        embedding_dim: int, 
        image_size: Tuple[int, int],
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.patch_size: int = patch_size
        self.embedding_dim: int = embedding_dim
        self.image_size: Tuple[int, int] = image_size
        self.n_hpatches: int = image_size[0] // patch_size
        self.n_wpatches: int = image_size[1] // patch_size
        self.n_patches: int = self.n_hpatches * self.n_wpatches
        self.projector = nn.Conv2d(
            in_channels=in_channels, out_channels=embedding_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4  # (batch_size, n_channels, height, width)
        batch_size: int = input.shape[0]
        output: torch.Tensor = self.projector(input)
        assert output.shape == (batch_size, self.embedding_dim, self.n_hpatches, self.n_wpatches)
        output: torch.Tensor = output.flatten(start_dim=2, end_dim=-1)
        assert output.shape == (batch_size, self.embedding_dim, self.n_patches)
        return output.permute(0, 2, 1)
    

class TransformerBlock(nn.Module):

    def __init__(
        self, 
        embedding_dim: int, 
        n_heads: int, 
        dropout: float,
    ):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_heads: int = n_heads

        assert embedding_dim % n_heads == 0, f'embedding_dim must be divisible by n_heads'
        self.head_embedding_dim: int = self.embedding_dim // self.n_heads
        
        self.qkv = nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 3)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=n_heads, 
            dropout=dropout, batch_first=False,
        )
        self.projector1 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.projector2 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 3
        assert input.shape[2] == self.embedding_dim
        batch_size: int = input.shape[0]
        n_patches: int = input.shape[1]

        residual: torch.Tensor = input.clone()
        
        # LayerNorm
        input: torch.Tensor = self.layer_norm1(input)
        
        # Multihead Attention
        qkv: torch.Tensor = self.qkv(input)
        assert qkv.shape == (batch_size, n_patches, self.embedding_dim * 3)
        qkv: torch.Tensor = qkv.reshape(batch_size, n_patches, 3, self.embedding_dim)
        qkv: torch.Tensor = qkv.permute(2, 1, 0, 3)
        assert qkv.shape == (3, n_patches, batch_size, self.embedding_dim)
        queries: torch.Tensor = qkv[0]
        keys: torch.Tensor = qkv[1]
        values: torch.Tensor = qkv[2]
        output, _ = self.attention(query=queries, key=keys, value=values)
        assert output.shape == (n_patches, batch_size, self.embedding_dim)
        output: torch.Tensor = output.permute(1, 0, 2)
        output = F.gelu(self.projector1(output))

        # Residual Connection
        output = residual + output
        residual: torch.Tensor = output.clone()
        # LayerNorm
        output = self.layer_norm2(output)
        # MLP
        output = F.gelu(self.projector2(output))
        # Residual Connection
        output = residual + output
        assert output.shape == (batch_size, n_patches, self.embedding_dim)
        return output


class TransformerEncoder(nn.Module):

    def __init__(
        self, 
        embedding_dim: int, 
        n_heads: int, 
        depth: int, 
        dropout: float
    ):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_heads: int = n_heads
        self.depth: int = depth
        self.dropout: float = dropout

        self.blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, n_heads, dropout) for _ in range(depth)]
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 3
        assert input.shape[2] == self.embedding_dim
        batch_size: int = input.shape[0]
        n_patches: int = input.shape[1]

        output: torch.Tensor = self.blocks(input)
        output: torch.Tensor = self.layer_norm(output)
        assert output.shape == (batch_size, n_patches, self.embedding_dim)
        return output


class OrthogonalLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size: int = input.shape[0]
        assert input.shape == (batch_size, 6, 2)
        s: torch.Tensor = - (input[:, 0, 0] - input[:, 1, 0]) / (input[:, 0, 1] - input[:, 1, 1])
        y3: torch.Tensor = s * (input[:, 3, 0] - input[:, 2, 0]) + input[:, 2, 1]
        output = input.clone()
        output[:, 3, 1] = y3
        assert output.shape == input.shape
        return output


class VisionTransformer(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        patch_size: int, 
        embedding_dim: int, 
        image_size: Tuple[int, int], 
        depth: int, 
        n_heads: int, 
        dropout: float, 
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = 12
        self.patch_size: int = patch_size
        self.embedding_dim: int = embedding_dim
        self.image_size: Tuple[int, int] = image_size
        self.depth: int = depth
        self.n_heads: int = n_heads
        self.dropout: float = dropout

        self.patch_embedding = PatchPositionEmbedding(in_channels, patch_size, embedding_dim, image_size)
        self.encoder = TransformerEncoder(embedding_dim, n_heads, depth, dropout)
        self.orthogonalizer = OrthogonalLayer()

        scale_pos: float = self.patch_embedding.n_patches * embedding_dim
        self.pos_embedding = nn.Parameter(
            data=torch.rand(1, self.patch_embedding.n_patches, embedding_dim) / scale_pos
        )
        scale_mlp: float = self.patch_embedding.n_patches * embedding_dim * self.out_channels
        self.mlp_head = nn.Sequential(*[
            nn.Linear(in_features=self.patch_embedding.n_patches * embedding_dim, out_features=1024), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(in_features=1024, out_features=512), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=self.out_channels),
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4
        batch_size, n_channels, image_height, image_width = input.shape
        output: torch.Tensor = self.patch_embedding(input)
        assert output.shape == (batch_size, self.patch_embedding.n_patches, self.embedding_dim)
        output: torch.Tensor = output + self.pos_embedding
        output: torch.Tensor = self.encoder(output)
        assert output.shape == (batch_size, self.patch_embedding.n_patches, self.embedding_dim)
        output: torch.Tensor = output.flatten(start_dim=1, end_dim=-1)
        output: torch.Tensor = self.mlp_head(output).reshape(batch_size, 6, 2)
        output: torch.Tensor = output.reshape(batch_size, 6, 2)
        return self.orthogonalizer(output)



if __name__ == '__main__':
    self = VisionTransformer(
        in_channels=1,
        patch_size=32,
        embedding_dim=256,
        image_size=(512, 512),
        depth=2,
        n_heads=16,
        dropout=0.1,
    )
    x = torch.rand(8, 1, 512, 512)
    y = self(x)
        






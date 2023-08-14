import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, dim=768):
        """
        This class is used to split images to patches
        :param img_size: the size of input images (B, C, H, W)
        :param patch_size:  the size of patchs (x, x), such as x = 16
        :param in_channels: the channels of input image
        :param dim: patches embedding dim = 16 * 16 * 3
        """
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = int(img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x : (B, C, H, W)
        x = self.proj(x)  # (B, dim, patch_size, patch_size)
        x = x.flatten(2)  # (B, dim, patch_size**2)
        x = x.transpose(1, 2)  # (B, patch_size**2, dim)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias, attn_p=0., proj_p=0.):
        """
        This is an implement of multi head attention
        :param dim: patches embedding dim = 16 * 16 * 3
        :param n_heads: the number of multi head
        :param qkv_bias: the bias of qkv linear layer
        :param attn_p: attn drop rate
        :param proj_p: projection drop rate
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim
        self.scale = (self.head_dim)**-0.5

        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop= nn.Dropout(attn_p)
        self.proj_drop= nn.Dropout(proj_p)

    def forward(self, x):
        # x : (B, N+1, D) B: batch_size, N: the number of patches, D: the dim of patch embedding
        b, n_patches, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)   # (B, N+1, 3*D)
        qkv = qkv.reshape(b, n_patches, 3, self.n_heads, self.head_dim)     # (B, N+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N+1, head_dim)

        q, k ,v = qkv[0], qkv[1], qkv[2]
        k_T = (k.transpose(-2, -1)) * self.scale  # (B, n_heads, head_dim, N+1)
        dp = q @ k_T                # (B, n_heads, N+1, N+1)
        attn = dp.softmax(dim=-1)   # (B, n_heads, N+1, N+1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v   # (B, n_heads, N+1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (B, N+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (B, N+1, dim)

        # mlp
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, p=0.):
        """
        MLP in Transformer Block
        :param dim: patch embedding dimension
        :param mlp_ratio: the hidden layer nodes expansion factor
        :param p: dropout rate
        """
        super(MLP, self).__init__()
        hidden_features = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, dim)
        self.drop = nn.Dropout(p)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.drop(self.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias, mlp_ratio=4.0, attn_p=0., proj_p=0., p=0.):
        """
        The Vision Transfomer Block
        :param dim: patches embedding dim = 16 * 16 * 3
        :param n_heads: the number of multi head
        :param qkv_bias: the bias of qkv linear layer
        :param attn_p: attn drop rate
        :param proj_p: projection drop rate
        :param mlp_ratio: the hidden layer nodes expansion factor
        :param p: dropout rate
        """
        super(Block, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mhsa = MultiHeadAttention(dim, n_heads, qkv_bias, attn_p, proj_p)
        self.mlp = MLP(dim, mlp_ratio, p)

    def forward(self, x):
        x = x + self.mhsa(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x

class ViT(nn.Module):
    def __init__(self,
                 img_size = 224,
                 patch_size = 16,
                 dim = 768,
                 in_channels = 3,
                 n_classes = 1000,
                 n_heads = 12,
                 depth = 12,
                 qkv_bias = True,
                 mlp_ratio = 4.0,
                 attn_p = 0.,
                 proj_p = 0.,
                 p = 0.
                 ):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embed = nn.Parameter(torch.zeros(1, 1+self.patch_embed.n_patches, dim))
        self.pos_drop = nn.Dropout(p)

        self.blocks = nn.ModuleList([
            Block(dim=dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p, p=p)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        b = x.shape[0]      # batch_size

        # patch embedding
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)   # (b, 197, dim)
        x += self.position_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        cls_token_output = x[:, 0]
        x = self.classifier(cls_token_output)
        return x

if __name__ == "__main__":
    patch_size = 16

    input = torch.randn(1, 3, 224, 224)
    patchembed = PatchEmbed(input.shape[2], patch_size, input.shape[1])
    print(patchembed(input).shape)
    # ==> torch.Size([1, 196, 768])

    patches_inputs = torch.randn(1, 197, 768)
    mhsa = MultiHeadAttention(768, 12, qkv_bias=True)
    print(mhsa(patches_inputs).shape)
    # ==> torch.Size([1, 197, 768])

    patches_inputs = torch.randn(1, 197, 768)
    block = Block(768, 12, True)
    print(block(patches_inputs).shape)
    # ==> torch.Size([1, 197, 768])

    input = torch.randn(1, 3, 224, 224)
    vit = ViT()
    print(vit(input).shape)
    # ==> torch.Size([1, 1000])


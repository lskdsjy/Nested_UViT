import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
import torch.nn.functional as F

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash' # 'fft' 'flash' '1-bit'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# Updated code lines

class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, use_checkpoint=False,
                 clip_dim=768, num_clip_token=77, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.context_embed = nn.Linear(clip_dim, embed_dim)

        self.extras = 1 + num_clip_token

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, context):
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        context_token = self.context_embed(context)
        x = torch.cat((time_token, context_token, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x

### NestedUViT with 1 block each

class NestedUViT_1(nn.Module):
    def __init__(self, checkpoint_path, **kwargs):
        super(NestedUViT_1, self).__init__()

        # Load pre-trained UViT model function
        def load_uvit():
            model = UViT(**kwargs)  # Initialize UViT with provided arguments
            checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load checkpoint
            state_dict = checkpoint.get("model", checkpoint)  # Get model state dict
            model.load_state_dict(state_dict, strict=True)  # Load the state dict into the model
            return model

        # Single Encoder Block
        self.encoder = load_uvit()

        # Bottleneck Block
        self.bottleneck = load_uvit()

        # Single Decoder Block
        self.decoder = load_uvit()

    def forward(self, x, timesteps, context, **kwargs):
        """
        Forward pass through the NestedUViT with 1 encoder, 1 bottleneck, and 1 decoder.
        
        Args:
            x: The input tensor
            timesteps: Time steps for the time embedding
            context: The context for the context embedding
            **kwargs: Additional arguments passed to the UViT blocks
        
        Returns:
            The final output from the decoder
        """

        # Encoder
        enc = self.encoder(x, timesteps, context, **kwargs)

        # Bottleneck
        bottleneck = self.bottleneck(enc, timesteps, context, **kwargs)

        # Decoder
        output = self.decoder(bottleneck + enc, timesteps, context, **kwargs)  # Skip connection from encoder

        return output

### NestedUViT with 2 blocks each

class NestedUViT_2(nn.Module):
    def __init__(self, checkpoint_path, **kwargs):
        super(NestedUViT_2, self).__init__()

        # Load pre-trained UViT model function
        def load_uvit():
            model = UViT(**kwargs)  # Initialize UViT with provided arguments
            checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load checkpoint
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint  # Get model state dict
            model.load_state_dict(state_dict, strict=True)  # Load the state dict into the model
            return model

        # Initialize nested UViT components
        self.encoder1 = load_uvit()  # First encoder block
        self.encoder2 = load_uvit()  # Second encoder block
        self.bottleneck = load_uvit()  # Bottleneck block
        self.decoder1 = load_uvit()  # First decoder block
        self.decoder2 = load_uvit()  # Second decoder block

    def forward(self, x, timesteps, context, **kwargs):
        """
        Forward pass through the nested UViT structure with skip connections.
        
        Args:
            x: The input tensor
            timesteps: Time steps for the time embedding
            context: The context for the context embedding
            **kwargs: Additional arguments passed to the UViT blocks
        
        Returns:
            The final output of the second decoder
        """

        # Encoder 1
        e1 = self.encoder1(x, timesteps, context, **kwargs)  # Pass through first encoder
        
        # Encoder 2
        e2 = self.encoder2(e1, timesteps, context, **kwargs)  # Pass through second encoder
        
        # Bottleneck
        b = self.bottleneck(e2, timesteps, context, **kwargs)  # Pass through bottleneck
        
        # First decoder (with skip connection from encoder 2)
        d2 = self.decoder1(b + e2, timesteps, context, **kwargs)  # Skip connection from encoder2 to decoder1
        
        # Second decoder (with skip connection from encoder 1)
        d1 = self.decoder2(d2 + e1, timesteps, context, **kwargs)  # Skip connection from encoder1 to decoder2

        return d1

### NestedUViT with 2 blocks each - concatenate side outputs

class NestedUViT_2S(nn.Module):
    def __init__(self, checkpoint_path, **kwargs):
        super(NestedUViT_2S, self).__init__()

        # Load pre-trained UViT model function
        def load_uvit():
            model = UViT(**kwargs)  # Initialize UViT with provided arguments
            checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load checkpoint
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint  # Get model state dict
            model.load_state_dict(state_dict, strict=True)  # Load the state dict into the model
            return model

        # Initialize nested UViT components
        self.encoder1 = load_uvit()  # First encoder block
        self.encoder2 = load_uvit()  # Second encoder block
        self.bottleneck = load_uvit()  # Bottleneck block
        self.decoder1 = load_uvit()  # First decoder block
        self.decoder2 = load_uvit()  # Second decoder block

    def forward(self, x, timesteps, context, **kwargs):
        """
        Forward pass through the nested UViT structure with skip connections.
        
        Args:
            x: The input tensor
            timesteps: Time steps for the time embedding
            context: The context for the context embedding
            **kwargs: Additional arguments passed to the UViT blocks
        
        Returns:
            The final output of the second decoder
        """
        # Encoders
        e1 = self.encoder1(x, timesteps, context, **kwargs)  # Pass through first encoder
        e2 = self.encoder2(e1, timesteps, context, **kwargs)  # Pass through second encoder
        
        # Bottleneck
        b = self.bottleneck(e2, timesteps, context, **kwargs)  # Pass through bottleneck
        
        # Decoders (with skip connection from encoder 2)
        d2 = self.decoder1(b + e2, timesteps, context, **kwargs)  # Skip connection from encoder2 to decoder1
        d1 = self.decoder2(d2 + e1, timesteps, context, **kwargs)  # Skip connection from encoder1 to decoder2
        
        # print("shape of d1:", d1.shape)
        output = d1+d2+b
        # # print("shape of output:", output.shape)
        return output

### NestedUViT with 3 blocks each

class NestedUViT_3(nn.Module):
    def __init__(self, checkpoint_path, **kwargs):
        super(NestedUViT_3, self).__init__()

        # Load pre-trained UViT model function
        def load_uvit():
            model = UViT(**kwargs)  # Initialize UViT with provided arguments
            checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load checkpoint
            state_dict = checkpoint.get("model", checkpoint)  # Get model state dict
            model.load_state_dict(state_dict, strict=True)  # Load the state dict into the model
            return model

        # Encoder Blocks
        self.encoder1 = load_uvit()
        self.encoder2 = load_uvit()
        self.encoder3 = load_uvit()

        # Bottleneck Block
        self.bottleneck = load_uvit()

        # Decoder Blocks
        self.decoder3 = load_uvit()
        self.decoder2 = load_uvit()
        self.decoder1 = load_uvit()

    def forward(self, x, timesteps, context, **kwargs):
        """
        Forward pass through the NestedUViT with 3 encoder, 1 bottleneck, and 3 decoder blocks.
        
        Args:
            x: The input tensor
            timesteps: Time steps for the time embedding
            context: The context for the context embedding
            **kwargs: Additional arguments passed to the UViT blocks
        
        Returns:
            The final output from the decoder
        """

        # Encoder stages
        e1 = self.encoder1(x, timesteps, context, **kwargs)
        e2 = self.encoder2(e1, timesteps, context, **kwargs)
        e3 = self.encoder3(e2, timesteps, context, **kwargs)

        # Bottleneck
        b = self.bottleneck(e3, timesteps, context, **kwargs)

        # Decoder stages with skip connections
        d3 = self.decoder3(b + e3, timesteps, context, **kwargs)  # Skip connection from e3
        d2 = self.decoder2(d3 + e2, timesteps, context, **kwargs)  # Skip connection from e2
        d1 = self.decoder1(d2 + e1, timesteps, context, **kwargs)  # Skip connection from e1

        return d1  # Final output


### NestedUViT with 4 blocks each

class NestedUViT_4(nn.Module):
    def __init__(self, checkpoint_path, **kwargs):
        super(NestedUViT_4, self).__init__()

        # Load pre-trained UViT model function
        def load_uvit():
            model = UViT(**kwargs)  # Initialize UViT with provided arguments
            checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load checkpoint
            state_dict = checkpoint.get("model", checkpoint)  # Get model state dict
            model.load_state_dict(state_dict, strict=True)  # Load the state dict into the model
            return model

        # Encoder Blocks
        self.encoder1 = load_uvit()
        self.encoder2 = load_uvit()
        self.encoder3 = load_uvit()
        self.encoder4 = load_uvit()

        # Bottleneck Block
        self.bottleneck = load_uvit()

        # Decoder Blocks
        self.decoder4 = load_uvit()
        self.decoder3 = load_uvit()
        self.decoder2 = load_uvit()
        self.decoder1 = load_uvit()

    def forward(self, x, timesteps, context, **kwargs):
        """
        Forward pass through the NestedUViT with 4 encoder, 1 bottleneck, and 4 decoder blocks.
        
        Args:
            x: The input tensor
            timesteps: Time steps for the time embedding
            context: The context for the context embedding
            **kwargs: Additional arguments passed to the UViT blocks
        
        Returns:
            The final output from the decoder
        """

        # Encoder stages
        e1 = self.encoder1(x, timesteps, context, **kwargs)
        e2 = self.encoder2(e1, timesteps, context, **kwargs)
        e3 = self.encoder3(e2, timesteps, context, **kwargs)
        e4 = self.encoder4(e3, timesteps, context, **kwargs)

        # Bottleneck
        b = self.bottleneck(e4, timesteps, context, **kwargs)

        # Decoder stages with skip connections
        d4 = self.decoder4(b + e4, timesteps, context, **kwargs)  # Skip connection from e4
        d3 = self.decoder3(d4 + e3, timesteps, context, **kwargs)  # Skip connection from e3
        d2 = self.decoder2(d3 + e2, timesteps, context, **kwargs)  # Skip connection from e2
        d1 = self.decoder1(d2 + e1, timesteps, context, **kwargs)  # Skip connection from e1

        return d1  # Final output



### UViT Stack with 3 blocks
class UViTStack(nn.Module):
    def __init__(self, checkpoint_path, **kwargs):
        super(UViTStack, self).__init__()

        # Load pre-trained UViT model function
        def load_uvit():
            model = UViT(**kwargs)  # Initialize UViT with provided arguments
            checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load checkpoint
            state_dict = checkpoint.get("model", checkpoint)  # Get model state dict
            model.load_state_dict(state_dict, strict=True)  # Load the state dict into the model
            return model

        # Stack of 3 UViT layers
        self.uvit_layers = nn.ModuleList([load_uvit() for _ in range(3)])

    def forward(self, x, timesteps, context, **kwargs):
        """
        Forward pass through the stacked UViT layers.
        
        Args:
            x: The input tensor
            timesteps: Time steps for the time embedding
            context: The context for the context embedding
            **kwargs: Additional arguments passed to the UViT layers
        
        Returns:
            The final output after passing through all UViT layers
        """
        for layer in self.uvit_layers:
            x = layer(x, timesteps, context, **kwargs)
        
        return x






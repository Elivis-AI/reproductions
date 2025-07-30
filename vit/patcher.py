import torch
import torch.nn as nn
from typing import Iterable
from functools import reduce
from einops import rearrange

class Patcher(nn.Module):
    """Patchify and unpatchify an image."""
    def __init__(self, image_shape: Iterable[int], patch_size: int, kh: int, kw: int):
        super().__init__()
        self.image_shape = image_shape  # (C, H, W)
        self.patch_size = patch_size
        self.kh = kh  # patches along height
        self.kw = kw  # patches along width

    @property
    def patch_shape(self):
        return (self.image_shape[0], self.patch_size, self.patch_size)

    @property
    def num_patch_elements(self):
        return reduce(lambda a, b: a * b, self.patch_shape)

    @property
    def num_patches(self):
        return self.kh * self.kw

    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        """Turn an image (possibly batched) into a collection of patches."""
        return rearrange(
            img,
            "... c (kh h) (kw w) -> ... (kh kw) c h w",
            h=self.patch_size,
            w=self.patch_size
        )

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Turn a collection of patches (possibly batched) back into an image."""
        return rearrange(
            patches,
            "... (kh kw) c h w -> ... c (kh h) (kw w)",
            kh=self.kh,
            kw=self.kw
        )

    def rasterize(self, patches: torch.Tensor) -> torch.Tensor:
        """Rasterize patches into tokens."""
        return rearrange(patches, "... c h w -> ... (c h w)")

    def unrasterize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Unrasterize tokens into patches."""
        c, h, w = self.patch_shape
        return rearrange(tokens, "... (c h w) -> ... c h w", c=c, h=h, w=w)

    def tokenify(self, img: torch.Tensor) -> torch.Tensor:
        """Turn img into rasterized patches."""
        return self.rasterize(self.patchify(img))

    def untokenify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Untokenify tokens into original image."""
        return self.unpatchify(self.unrasterize(tokens))

    def patchified_shape(self):
        """The expected shape of a patchified image."""
        return (self.num_patches, *self.patch_shape)

    @classmethod
    def from_img(cls, img: torch.Tensor, patch_size: int):
        """Create a Patcher from an example image."""
        return cls.from_img_shape(img.shape, patch_size)

    @classmethod
    def from_img_shape(cls, img_shape: Iterable[int], patch_size: int):
        """Create a patcher from a specified image shape."""
        height, width = img_shape[-2:]
        assert height % patch_size == 0, "Image height must be divisible by patch size"
        assert width % patch_size == 0, "Image width must be divisible by patch size"
        kh = height // patch_size
        kw = width // patch_size
        return cls(img_shape, patch_size, kh, kw)

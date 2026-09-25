import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional, Callable, Union, Tuple, Sequence
from .misc import GRN, LayerNorm2d

class ConvNeXt2Block2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion: int,
        kernel_size: int,
        use_grn: bool = True,
        norm: str = "layer",
        num_groups: int | None = None,
    ):
        super().__init__()

        hidden_channels = expansion * in_channels
        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )

        if norm == "layer":
            self.norm = LayerNorm2d(in_channels)

        elif norm == "group":
            if num_groups is None:
                raise ValueError(
                    "num_groups must be specified when norm='group'."
                )

            if in_channels % num_groups != 0:
                raise ValueError(
                    f"in_channels ({in_channels}) must be divisible "
                    f"by num_groups ({num_groups})."
                )

            self.norm = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=in_channels,
            )

        elif norm == "instance":
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels,
            )

        elif norm == "none":
            self.norm = nn.Identity()

        else:
            raise ValueError(
                f"Unknown norm '{norm}'. "
                "Choose from: 'layer', 'group', 'instance', 'none'."
            )

        self.expand = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.activation = nn.GELU()

        self.grn = (
            GRN(hidden_channels, 2)
            if use_grn
            else nn.Identity()
        )

        self.compress = nn.Conv2d(
            hidden_channels,
            in_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.compress(x)

        return residual + x
class ConvNeXt2BlockStack2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        expansion: int,
        kernel_size: int,
        use_grn: bool,
        norm: str = "layer",
        num_groups: int | None = None,
    ):
        super().__init__(
            *[
                ConvNeXt2Block2d(
                    in_channels=in_channels,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    use_grn=use_grn,
                    norm=norm,
                    num_groups=num_groups
                )
                for _ in range(depth)
            ]
        )
class ConvNeXt2Down2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        norm: str = "layer",
        num_groups: int | None = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = 2 * in_channels

        if norm == "layer":
            self.norm = LayerNorm2d(in_channels)

        elif norm == "group":
            if num_groups is None:
                raise ValueError("num_groups must be specified when norm='group'.")

            if in_channels % num_groups != 0:
                raise ValueError(
                    f"in_channels ({in_channels}) must be divisible "
                    f"by num_groups ({num_groups})."
                )

            self.norm = nn.GroupNorm(num_groups, in_channels)

        elif norm == "none":
            self.norm = nn.Identity()

        else:
            raise ValueError(f"Unknown norm '{norm}'.")

        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.norm(x))
class ConvNeXt2Up2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels // 2

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

class MedNeXtBlock3d(nn.Module):
    def __init__(self, channels : int, expansion : int, kernel_size : int, use_grn : bool = True):
        super().__init__()
        hidden_channels = expansion * channels
        padding = kernel_size // 2

        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )

        self.norm = nn.GroupNorm(
            num_groups=channels,
            num_channels=channels,
        )

        self.expand = nn.Conv3d(
            channels,
            hidden_channels,
            kernel_size=1,
        )

        self.activation = nn.GELU()
        self.grn = GRN(hidden_channels, 3) if use_grn else nn.Identity()

        self.compress = nn.Conv3d(
            hidden_channels,
            channels,
            kernel_size=1,
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.compress(x)

        return residual + x
class MedNeXtDown3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        kernel_size: int,
        use_grn: bool = True,
    ):
        super().__init__()

        hidden_channels = expansion * in_channels

        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.expand = nn.Conv3d(in_channels, hidden_channels, 1)
        self.activation = nn.GELU()
        self.grn = GRN(hidden_channels, 3) if use_grn else nn.Identity()
        self.compress = nn.Conv3d(hidden_channels, out_channels, 1)

        self.residual = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.compress(x)

        return x + residual
class MedNeXtUp3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        kernel_size: int = 5,
        use_grn: bool = True,
    ):
        super().__init__()

        hidden_channels = expansion * in_channels

        # output_padding=1 produces exact 2x spatial upsampling
        # for odd kernels and padding=kernel_size // 2.
        self.depthwise = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
            groups=in_channels,
        )

        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.expand = nn.Conv3d(in_channels, hidden_channels, 1)
        self.activation = nn.GELU()
        self.grn = GRN(hidden_channels, 3) if use_grn else nn.Identity()
        self.compress = nn.Conv3d(hidden_channels, out_channels, 1)

        self.residual = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.compress(x)

        return x + residual
class MedNeXtBlockStack3d(nn.Sequential):
    def __init__(
        self,
        channels: int,
        depth: int,
        expansion: int,
        kernel_size: int,
        use_grn: bool,
    ):
        super().__init__(
            *[
                MedNeXtBlock3d(
                    channels=channels,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    use_grn=use_grn,
                )
                for _ in range(depth)
            ]
        )
class MedNeXt3d(nn.Module):
    """
    Generalized MedNeXt-style 3D encoder-decoder.

    If channel_multipliers has length L + 1, the network contains:

        L encoder stages
        1 bottleneck stage
        L decoder stages

    Therefore depths and expansions must each contain 2L + 1 entries.

    Example
    -------
    channel_multipliers = (1, 2, 4, 8)

    gives:

        encoder widths:  C, 2C, 4C
        bottleneck:      8C
        decoder widths:  4C, 2C, C

    and requires:

        len(depths) == len(expansions) == 7
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        kernel_size: int,
        channel_multipliers: Sequence[int],
        depths: Sequence[int] | None = None,
        expansions: Sequence[int] | None = None,
        use_grn: bool = True,
    ):
        super().__init__()

        channel_multipliers = tuple(channel_multipliers)

        if len(channel_multipliers) < 2:
            raise ValueError(
                "channel_multipliers must contain at least two entries: "
                "one encoder width and one bottleneck width."
            )

        if any(multiplier <= 0 for multiplier in channel_multipliers):
            raise ValueError("All channel multipliers must be positive.")

        self.num_levels = len(channel_multipliers) - 1
        num_stages = 2 * self.num_levels + 1

        # Default to two blocks and expansion factor four at every stage.
        if depths is None:
            depths = (2,) * num_stages

        if expansions is None:
            expansions = (4,) * num_stages

        depths = tuple(depths)
        expansions = tuple(expansions)

        if len(depths) != num_stages:
            raise ValueError(
                f"Expected {num_stages} depth values for "
                f"{self.num_levels} encoder levels, but received "
                f"{len(depths)}."
            )

        if len(expansions) != num_stages:
            raise ValueError(
                f"Expected {num_stages} expansion values for "
                f"{self.num_levels} encoder levels, but received "
                f"{len(expansions)}."
            )

        if any(depth < 0 for depth in depths):
            raise ValueError("Depth values must be non-negative.")

        if any(expansion <= 0 for expansion in expansions):
            raise ValueError("Expansion values must be positive.")

        widths = tuple(
            base_channels * multiplier
            for multiplier in channel_multipliers
        )

        self.widths = widths
        self.depths = depths
        self.expansions = expansions

        self.stem = nn.Conv3d(
            in_channels,
            widths[0],
            kernel_size=1,
        )

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for level in range(self.num_levels):
            self.encoders.append(
                MedNeXtBlockStack3d(
                    channels=widths[level],
                    depth=depths[level],
                    expansion=expansions[level],
                    kernel_size=kernel_size,
                    use_grn=use_grn,
                )
            )

            self.downsamplers.append(
                MedNeXtDown3d(
                    in_channels=widths[level],
                    out_channels=widths[level + 1],
                    expansion=expansions[level],
                    kernel_size=kernel_size,
                    use_grn=use_grn,
                )
            )

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------

        bottleneck_index = self.num_levels

        self.bottleneck = MedNeXtBlockStack3d(
            channels=widths[-1],
            depth=depths[bottleneck_index],
            expansion=expansions[bottleneck_index],
            kernel_size=kernel_size,
            use_grn=use_grn,
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()

        decoder_levels = tuple(
            reversed(range(self.num_levels))
        )

        for decoder_offset, level in enumerate(decoder_levels):
            stage_index = self.num_levels + 1 + decoder_offset

            self.upsamplers.append(
                MedNeXtUp3d(
                    in_channels=widths[level + 1],
                    out_channels=widths[level],
                    expansion=expansions[stage_index],
                    kernel_size=kernel_size,
                    use_grn=use_grn,
                )
            )

            self.decoders.append(
                MedNeXtBlockStack3d(
                    channels=widths[level],
                    depth=depths[stage_index],
                    expansion=expansions[stage_index],
                    kernel_size=kernel_size,
                    use_grn=use_grn,
                )
            )

        self.output_head = nn.Conv3d(
            widths[0],
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "Expected input shape [B, C, Z, Y, X], "
                f"but received {tuple(x.shape)}."
            )

        x = self.stem(x)

        skips: list[torch.Tensor] = []

        for encoder, downsample in zip(
            self.encoders,
            self.downsamplers,
        ):
            x = encoder(x)
            skips.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(
            self.upsamplers,
            self.decoders,
            reversed(skips),
        ):
            x = upsample(x)

            # Handles odd dimensions without assuming divisibility by 2^L.
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )

            # MedNeXt uses additive skip connections.
            x = x + skip
            x = decoder(x)

        return self.output_head(x)

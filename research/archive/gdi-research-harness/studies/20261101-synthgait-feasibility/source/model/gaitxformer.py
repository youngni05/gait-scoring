import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for the gait query tokens."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # The final checkpoint writer saved learnable parameters only. Keeping
        # this deterministic buffer non-persistent makes strict loading match
        # that checkpoint format without changing model outputs.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class CrossAttentionLayer(nn.Module):
    """Decoder block used by the final cross-attention checkpoint."""

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, memory):
        attn_out, _ = self.cross_attn(query, memory, memory)
        query = self.norm1(query + self.dropout1(attn_out))
        ffn_out = self.ffn(query)
        return self.norm2(query + self.dropout2(ffn_out))


class GaitXFormer(nn.Module):
    """Final V-JEPA2 + cross-attention gait-regression architecture."""

    def __init__(
        self,
        encoder_name="vjepa2_vit_large",
        n_decoder_layers=3,
        expansion_ratio=2,
        n_out=6,
        patch_t=2,
        patch_h=16,
        patch_w=16,
        decoder_type="cross_attn",
        encoder_freeze_strategy="full_finetune",
        camera_tokens=False,
    ):
        super().__init__()
        if decoder_type != "cross_attn":
            raise ValueError("The cleaned model supports only decoder_type='cross_attn'.")
        if encoder_freeze_strategy not in {"full_finetune", "freeze_encoder"}:
            raise ValueError(
                "The cleaned model supports only full_finetune and freeze_encoder."
            )
        if camera_tokens:
            raise ValueError("Camera tokens are not part of the retained final model.")

        self.encoder = torch.hub.load("facebookresearch/vjepa2", encoder_name)[0]
        self.encoder_dim = 1024
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.decoder_type = decoder_type

        requires_grad = encoder_freeze_strategy == "full_finetune"
        for parameter in self.encoder.parameters():
            parameter.requires_grad = requires_grad

        trainable_params = sum(
            parameter.numel()
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        )
        total_params = sum(parameter.numel() for parameter in self.encoder.parameters())
        print(f"[INFO] Encoder freezing strategy: {encoder_freeze_strategy}")
        print(
            "[INFO] Trainable encoder params: "
            f"{trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        self.gait_tokens = nn.Parameter(torch.randn(1, n_out, self.encoder_dim))
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    d_model=self.encoder_dim,
                    nhead=8,
                    dim_feedforward=self.encoder_dim * expansion_ratio,
                    dropout=0.1,
                )
                for _ in range(n_decoder_layers)
            ]
        )
        self.gait_heads = nn.ModuleList(
            [nn.Linear(self.encoder_dim, 1) for _ in range(n_out)]
        )
        self.query_pos_encoding = PositionalEncoding(
            self.encoder_dim,
            max_len=n_out,
        )

    def forward(self, x, valid=None, return_features=False):
        del valid  # Padding is represented in the cached video tensor as in the final run.
        batch_size = x.shape[0]
        features = self.encoder(x)

        decoded_features = self.query_pos_encoding(
            self.gait_tokens.expand(batch_size, -1, -1)
        )
        for layer in self.cross_attn_layers:
            decoded_features = layer(decoded_features, features)

        feature_embedding = decoded_features.mean(dim=1)
        output = torch.cat(
            [
                head(decoded_features[:, index, :])
                for index, head in enumerate(self.gait_heads)
            ],
            dim=1,
        )
        if return_features:
            return output, feature_embedding
        return output

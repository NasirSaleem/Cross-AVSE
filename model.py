import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchinfo import summary  # Add this import

class DepthConv1d(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True):
        super().__init__()
        self.skip = skip
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel,
                                 dilation=dilation, groups=hidden_channel,
                                 padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        return residual


class MultiLayerCrossAttention(nn.Module):
    def __init__(self, input_size, layer, in_ch, kernel_size, dilation):
        super().__init__()
        self.audio_encoder = nn.ModuleList()
        self.visual_encoder = nn.ModuleList()
        self.layer = layer
        self.projection = nn.Conv1d(in_ch * 2, in_ch, kernel_size, padding='same')  # Changed from 4 to 2
        self.LayernormList_audio = nn.ModuleList()
        self.LayernormList_visual = nn.ModuleList()
        self.layernorm_out = nn.GroupNorm(1, in_ch, eps=1e-08)

        for _ in range(layer):
            self.LayernormList_audio.append(nn.GroupNorm(1, in_ch, eps=1e-08))
            self.LayernormList_visual.append(nn.GroupNorm(1, in_ch, eps=1e-08))
            self.audio_encoder.append(DepthConv1d(in_ch, in_ch, kernel_size, padding='same'))  # Removed *2
            self.visual_encoder.append(DepthConv1d(in_ch, in_ch, kernel_size, padding='same'))  # Removed *2

    def forward(self, audio, visual):
        out_audio = audio.permute(0, 2, 1)  # [B, C, T]
        out_visual = visual.permute(0, 2, 1)  # [B, C, T]

        residual_audio = out_audio
        residual_visual = out_visual

        for i in range(self.layer):
            # Process audio and visual streams
            audio_out = self.audio_encoder[i](out_audio)
            visual_out = self.visual_encoder[i](out_visual)

            # Handle tuple outputs
            if isinstance(audio_out, tuple):
                audio_out = audio_out[0]  # Take residual
            if isinstance(visual_out, tuple):
                visual_out = visual_out[0]  # Take residual

            # Residual connections
            out_audio = self.LayernormList_audio[i](audio_out + residual_audio)
            out_visual = self.LayernormList_visual[i](visual_out + residual_visual)

            residual_audio = out_audio
            residual_visual = out_visual

        # Simplified projection
        out = torch.cat((out_audio, out_visual), dim=1)  # [B, 2*C, T]
        out = self.projection(out)  # [B, C, T]
        return self.layernorm_out(out).permute(0, 2, 1)  # [B, T, C]


class LightVisualEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v3_small(weights=weights)
        self.features = base_model.features
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(3),
            nn.Hardswish()
        )
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 128),  # Reduced to 128 to match audio
            nn.LayerNorm(128),
            nn.Hardswish()
        )
        if pretrained:
            for param in list(self.features.children())[:4]:
                param.requires_grad = False

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.temporal_conv(x)
        visual_features = []
        for t in range(T):
            frame = x[:, :, t]
            features = self.features(frame)
            projected = self.projection(features)
            visual_features.append(projected)
        return torch.stack(visual_features, dim=1)  # [B, T, 128]


class AudioEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, kernel_size=16):  # Reduced to 128
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, groups=4),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x.unsqueeze(1))  # [B, 128, T]


class AudioDecoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=1, kernel_size=16, stride=8):  # Matched to 128
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, bias=False
        )

    def forward(self, x):
        return self.deconv(x).squeeze(1)


class Separator(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, hidden_channels=128, num_layers=6):  # Unified to 128
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.PReLU(),
            nn.GroupNorm(1, out_channels)
        )
        self.blocks = nn.ModuleList([
            DepthConv1d(out_channels, hidden_channels, 3, padding=1)
            for _ in range(num_layers)
        ])
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        skip = 0
        for block in self.blocks:
            res, sk = block(x)
            x = x + res
            skip = skip + sk
        return self.output(skip) * self.output_gate(skip)


class AVSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder(out_channels=128)
        self.visual_encoder = LightVisualEncoder()
        self.cross_attention = MultiLayerCrossAttention(
            input_size=128,
            layer=3,
            in_ch=128,
            kernel_size=3,
            dilation=1
        )
        self.separator = Separator(in_channels=128, out_channels=128)
        self.audio_decoder = AudioDecoder(in_channels=128)

    def forward(self, inputs):
        audio_feats = self.audio_encoder(inputs["noisy_audio"])  # [B, 128, T]
        visual_feats = self.visual_encoder(inputs["video_frames"])  # [B, T, 128]

        # Align temporal dimensions
        B, C, T = audio_feats.shape
        visual_feats = F.interpolate(visual_feats.permute(0, 2, 1), size=T, mode='linear').permute(0, 2, 1)

        # Cross attention
        fused_feats = self.cross_attention(
            audio_feats.permute(0, 2, 1),  # [B, T, 128]
            visual_feats  # [B, T, 128]
        )  # [B, T, 128]

        # Separation
        mask = self.separator(fused_feats.permute(0, 2, 1))  # [B, 128, T]

        # Apply mask
        out = mask * audio_feats  # [B, 128, T]

        return self.audio_decoder(out)  # [B, L]


class AVSEModule(LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-3, val_dataset=None, lips=False, rgb=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = AVSE()
        self.val_dataset = val_dataset

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        clean = batch["clean"]
        enhanced = self(batch)
        loss = -self._cal_si_snr(enhanced, clean)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        clean = batch["clean"]
        enhanced = self(batch)
        loss = -self._cal_si_snr(enhanced, clean)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _cal_si_snr(self, est, target):
        eps = 1e-8
        target = target - target.mean(dim=1, keepdim=True)
        est = est - est.mean(dim=1, keepdim=True)

        s_target = (target * est).sum(dim=1, keepdim=True) * target / \
                   (target.pow(2).sum(dim=1, keepdim=True) + eps)
        e_noise = est - s_target

        si_snr = 10 * torch.log10(
            s_target.pow(2).sum(dim=1) / (e_noise.pow(2).sum(dim=1) + eps))
        return si_snr.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def enhance(self, data):
        inputs = {
            "noisy_audio": torch.from_numpy(data["noisy_audio"][None]).float().to(self.device),
            "video_frames": torch.from_numpy(data["video_frames"][None]).float().to(self.device)
        }
        with torch.no_grad():
            enhanced = self(inputs).cpu().numpy()[0]
        return enhanced / (np.abs(enhanced).max() + 1e-7)

    def print_model_summary(self, sample_length=16000, video_frames=30):
        """Print comprehensive model summary with input/output dimensions"""
        dummy_input = {
            "noisy_audio": torch.rand(2, sample_length),  # batch of 2
            "video_frames": torch.rand(2, 1 if not self.hparams.get('rgb', False) else 3,
                                       video_frames,
                                       96 if self.hparams.get('lips', False) else 224,
                                       96 if self.hparams.get('lips', False) else 224)
        }

        summary(
            self.model,
            input_data=dummy_input,
            depth=5,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            verbose=1
        )
import torch
from torchsummary import summary
from model import AVSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AVSE().to(device)

# Print detailed summary
print("\n" + "="*80)
print("AVSE MODEL DETAILED SUMMARY")
print("="*80 + "\n")

# 1. Audio Encoder
print("1. AUDIO ENCODER")
print("-"*50)
summary(model.audio_encoder, input_size=(1, 16000), device=device.type)
print("\n" + "-"*50)
print(f"Input:  Noisy audio waveform [B, 1, 16000]")
print(f"Output: Audio features [B, 256, T] (T depends on input length)")
print("Process: 1D convolution with kernel=16, stride=8 → BatchNorm → GELU")
print("="*80 + "\n")

# 2. Visual Encoder
print("2. VISUAL ENCODER")
print("-"*50)
summary(model.visual_encoder, input_size=(1, 75, 224, 224), device=device.type)
print("\n" + "-"*50)
print(f"Input:  Video frames [B, 1, 75, 224, 224] (B×T×C×H×W)")
print(f"Output: Visual features [B, 75, 256]")
print("Process:")
print("- Temporal Conv3D (1→3 channels)")
print("- MobileNetV3-small feature extraction (frozen first 4 layers)")
print("- Adaptive pooling + projection to 256D")
print("="*80 + "\n")

# 3. Cross Attention Module
print("3. CROSS ATTENTION MODULE")
print("-"*50)
print(f"Input:  Audio [B, T, 256], Visual [B, T, 256]")
print(f"Output: Fused features [B, T, 256]")
print("\nComponents:")
for i, block in enumerate(model.cross_attention.audio_encoder):
    print(f"  Layer {i+1}:")
    print(f"  - Audio Encoder: DepthConv1d(256→512→256)")
    print(f"  - Visual Encoder: DepthConv1d(256→512→256)")
    print(f"  - GroupNorm + Residual connections")
print("\nFinal Projection: Conv1d(1024→256)")
print("="*80 + "\n")

# 4. Separator
print("4. SEPARATOR")
print("-"*50)
summary(model.separator, input_size=(512, 100), device=device.type)  # Example T=100
print("\n" + "-"*50)
print(f"Input:  Fused features [B, 512, T]")
print(f"Output: Mask [B, 128, T]")
print("Process:")
print("- Initial Conv1d(512→128)")
print("- 6x DepthConv1d blocks with skip connections")
print("- Output gates (Tanh * Sigmoid)")
print("="*80 + "\n")

# 5. Audio Decoder
print("5. AUDIO DECODER")
print("-"*50)
summary(model.audio_decoder, input_size=(256, 100), device=device.type)  # Example T=100
print("\n" + "-"*50)
print(f"Input:  Masked features [B, 256, T]")
print(f"Output: Enhanced waveform [B, L] (L depends on input length)")
print("Process: ConvTranspose1d with kernel=16, stride=8")
print("="*80 + "\n")

# Full Model Statistics
print("FULL MODEL STATISTICS")
print("-"*50)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {total_params - trainable_params:,}")
print("\nKey Components:")
print("- Audio Encoder: ~300K params")
print("- Visual Encoder: ~1.5M params (mostly frozen)")
print("- Cross Attention: ~2M params")
print("- Separator: ~1.2M params")
print("- Audio Decoder: ~100K params")
print("="*80 + "\n")

# Input/Output Flow
print("INPUT/OUTPUT FLOW")
print("-"*50)
print("1. Input Audio: [B, 1, 16000]")
print("2. Audio Encoder → [B, 256, T]")
print("3. Input Video: [B, 1, 75, 224, 224]")
print("4. Visual Encoder → [B, 75, 256]")
print("5. Temporal Interpolation → Match audio T")
print("6. Cross Attention → [B, T, 256]")
print("7. Separator → [B, 128, T]")
print("8. Apply mask to audio features")
print("9. Audio Decoder → [B, L]")
print("="*80)
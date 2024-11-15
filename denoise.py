import os
import torch
import torchaudio
import soundfile as sf
from gtcrn import GTCRN
from tqdm import tqdm

# Load model
checkpoint_path = 'exp/logs/gtcrn_28spk/3a/checkpoints/epoch=17-step=2088.ckpt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = GTCRN().to(device).eval()
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)

# Define function to process each audio file
def denoise_audio(file_path, model):
    # Load noisy audio file
    noisy_waveform, sr = torchaudio.load(file_path)
    assert sr == 16000, "Sample rate should be 16kHz."

    # Move waveform to the same device as the model
    noisy_waveform = noisy_waveform.to(device)

    # Convert waveform to spectrogram
    noisy_spectrogram = torch.stft(
        noisy_waveform.squeeze(0), 
        n_fft=512, hop_length=256, 
        win_length=512, 
        window=torch.hann_window(512, device=device),  # Ensure window is on the correct device
        return_complex=True
    )
    noisy_spectrogram = torch.view_as_real(noisy_spectrogram)
    # Pass through model
    with torch.no_grad():
        enhanced_spectrogram = model(noisy_spectrogram.unsqueeze(0))[0]

    # Convert output to complex format for ISTFT
    enhanced_complex = torch.complex(
        enhanced_spectrogram[..., 0], enhanced_spectrogram[..., 1]
    )

    # Inverse STFT to waveform
    enhanced_waveform = torch.istft(
        enhanced_complex, n_fft=512, hop_length=256, 
        win_length=512, window=torch.hann_window(512, device=device)
    )

    return enhanced_waveform.cpu(), sr

# Directory paths
input_dir = 'VCTK-DEMAND/test/noisy'
output_dir = 'VCTK-DEMAND/test/enh_3a'
os.makedirs(output_dir, exist_ok=True)

# Process all files in input directory
for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
    if filename.endswith('.wav'):
        file_path = os.path.join(input_dir, filename)
        enhanced_waveform, sr = denoise_audio(file_path, model)

        # Save enhanced waveform
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, enhanced_waveform.numpy(), sr)

print("All files processed.")
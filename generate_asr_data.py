import os
import json
import torch
import torchaudio
import numpy as np
import librosa
from transformers import AutoTokenizer
from TTS.api import TTS
import torchaudio.transforms as T
import torchaudio.functional as F
from torchaudio.utils import download_asset
from datasets import load_dataset

# Models.
TTS_MODEL = "tts_models/en/ljspeech/fast_pitch"
LLM_MODEL = "gpt2"

# Load TTS model
tts = TTS(model_name=TTS_MODEL).to("mps")

# Load LLM BPE tokenizer (e.g., GPT-2)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
max_seq_len = tokenizer.model_max_length

# Load audio samples for data augmentation.

# Room Impulse Response (RIR) for reverberation.
SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
# Clean up the RIR. Extract the main impulse and normalize it by its power.
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.linalg.vector_norm(rir, ord=2)

# Background noise for SNR.
#noise_file="background_noise.wav"
#noise_waveform, _ = librosa.load(noise_file, sr=sample_rate)
#SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
#noise, _ = torchaudio.load(SAMPLE_NOISE)
#snr_dbs = torch.tensor([20, 10, 3])


# Define augmentation functions
def speed_perturb(waveform, sample_rate, factor):
    """Apply speed perturbation by resampling."""
    new_sample_rate = int(sample_rate * factor)
    return torchaudio.functional.resample(waveform, sample_rate, new_sample_rate), new_sample_rate

def pitch_shift(waveform, sample_rate, semitones):
    """Shift pitch up or down by semitones."""
    return F.pitch_shift(waveform, sample_rate, semitones)

def add_noise(waveform, noise_level=0.02):
    """Add Gaussian noise to the waveform."""
    noise = np.random.normal(0, noise_level, waveform.shape)
    return waveform + torch.tensor(noise, dtype=waveform.dtype)

def add_background_noise(waveform, sample_rate, noise_waveform, noise_level=0.02):
    """Add real background noise from a file."""
    noise_waveform = torch.tensor(noise_waveform[:waveform.shape[-1]]) * noise_level
    return waveform + noise_waveform

def apply_spec_augment(mel_spectrogram):
    """Apply SpecAugment to mask parts of the spectrogram."""
    freq_mask = T.FrequencyMasking(freq_mask_param=15)
    time_mask = T.TimeMasking(time_mask_param=50)
    return time_mask(freq_mask(mel_spectrogram))


def augment_audio(waveform, sample_rate):
    """Apply a series of augmentations and return a list of augmented versions."""
    augmented_audios = []

    # Speed perturbation
    augmented_audios.append(speed_perturb(waveform, sample_rate, 1.1)[0])  # Faster
    augmented_audios.append(speed_perturb(waveform, sample_rate, 0.9)[0])  # Slower

    # Pitch shift
    augmented_audios.append(pitch_shift(waveform, sample_rate, 2))  # Higher pitch
    augmented_audios.append(pitch_shift(waveform, sample_rate, -2)) # Lower pitch

    # Gaussian noise
    augmented_audios.append(add_noise(waveform, 0.01))
    # Real world noise
    #augmented_audios.append(add_background_noise(waveform, sample_rate, 
    #                                             noise_waveform=noise_waveform, 
    #                                             noise_level=0.01))
    
    # Noise again...
    #noise_scaled = noise[:, : waveform.shape[1]]
    #noisy_speeches = F.add_noise(waveform, noise_scaled, snr_dbs)
    #augmented_audios += noisy_speeches.tolist()
    
    # Reverberation
    reverb = F.fftconvolve(waveform, rir)
    augmented_audios.append(reverb)

    return augmented_audios

# Create dataset folder
os.makedirs("asr_dataset", exist_ok=True)

# Load dataset (you can replace this with your own text corpus)
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:100]")  # First 1K samples

# Data processing loop
dataset_jsonl = "asr_dataset/synthetic_asr_data.jsonl"

with open(dataset_jsonl, "w") as f:
    for i, sample in enumerate(dataset):
        text = sample["text"][:max_seq_len]
        wav_path = f"asr_dataset/audio_{i}.wav"

        # Generate speech
        tts.tts_to_file(text=text, file_path=wav_path)

        # Load waveform
        waveform, sample_rate = torchaudio.load(wav_path)

        # Convert text to BPE tokens
        bpe_tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0).tolist()

        # Save original sample
        entry = {"audio_path": wav_path, "bpe_tokens": bpe_tokens, "transcript": text}
        f.write(json.dumps(entry) + "\n")

        # Augment data
        augmented_waves = augment_audio(waveform, sample_rate)

        for j, aug_wave in enumerate(augmented_waves):
            aug_path = f"asr_dataset/audio_{i}_aug_{j}.wav"
            torchaudio.save(aug_path, aug_wave, sample_rate)

            aug_entry = {"audio_path": aug_path, "bpe_tokens": bpe_tokens, "transcript": text}
            f.write(json.dumps(aug_entry) + "\n")

        print(f"Processed {i+1}/{len(dataset)} samples...", end="\r")

print("\nData generation complete! Files saved in 'asr_dataset/'")

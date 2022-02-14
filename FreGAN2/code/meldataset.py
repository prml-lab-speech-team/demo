import math
import os
import random
import torch
import torch.utils.data
from torch.nn import AvgPool1d
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from pathlib import Path
import glob

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist_custom(a):
    training_files = glob.glob(os.path.join(a.input_training_file, '*-wave.npy'), recursive=True)
    validation_files = glob.glob(os.path.join(a.input_validation_file, '*-wave.npy'), recursive=True)
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.device = device

    def __getitem__(self, index):
        audio_filename = self.audio_files[index]
        mel_filename = audio_filename.replace('wave', 'feats')

        audio = np.load(audio_filename)
        audio = torch.FloatTensor(audio)  # [1, T]
        audio = audio.unsqueeze(0)

        mel = np.load(mel_filename)
        mel = torch.from_numpy(mel).unsqueeze(0).transpose(-1, -2)

        if self.split:
            frames_per_seg = math.ceil(self.segment_size / self.hop_size)

            if audio.size(1) > self.segment_size:
                mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
            else:
                mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), audio_filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)

# import math
# import os
# import random
# import torch
# import torch.utils.data
# from torch.nn import AvgPool1d
# import numpy as np
# from librosa.util import normalize
# from scipy.io.wavfile import read
# from librosa.filters import mel as librosa_mel_fn
# from pathlib import Path
# import glob
#
# MAX_WAV_VALUE = 32768.0
#
#
# def load_wav(full_path):
#     sampling_rate, data = read(full_path)
#     return data, sampling_rate
#
#
# def dynamic_range_compression(x, C=1, clip_val=1e-5):
#     return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)
#
#
# def dynamic_range_decompression(x, C=1):
#     return np.exp(x) / C
#
#
# def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
#     return torch.log(torch.clamp(x, min=clip_val) * C)
#
#
# def dynamic_range_decompression_torch(x, C=1):
#     return torch.exp(x) / C
#
#
# def spectral_normalize_torch(magnitudes):
#     output = dynamic_range_compression_torch(magnitudes)
#     return output
#
#
# def spectral_de_normalize_torch(magnitudes):
#     output = dynamic_range_decompression_torch(magnitudes)
#     return output
#
#
# mel_basis = {}
# hann_window = {}
#
# def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
#     # global mel_basis, hann_window
#     # if fmax not in mel_basis:
#     #     mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
#     #     mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
#     #     hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
#     #
#     # y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
#     #                             mode='reflect')
#     # y = y.squeeze(1)
#     #
#     # spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
#     #                   center=center, pad_mode='reflect', normalized=False, onesided=True)
#     #
#     # spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
#     #
#     # spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
#     # spec = spectral_normalize_torch(spec)
#     #
#     # return spec
#
#     if torch.min(y) < -1.:
#         print('min value is ', torch.min(y))
#     if torch.max(y) > 1.:
#         print('max value is ', torch.max(y))
#
#     global mel_basis, hann_window
#     if fmax not in mel_basis:
#         mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
#         mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
#         hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
#
#     y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
#                                 mode='reflect')
#     y = y.squeeze(1)
#
#     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
#                       center=center, pad_mode='reflect', normalized=False, onesided=True)
#
#     spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
#
#     spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
#     spec = spectral_normalize_torch(spec)
#
#     return spec
#
# def get_dataset_filelist_custom(a):
#     training_files = glob.glob(os.path.join(a.input_training_file, '*-wave.npy'), recursive=True)
#     validation_files = glob.glob(os.path.join(a.input_validation_file, '*-wave.npy'), recursive=True)
#     return training_files, validation_files
#
#
# class MelDataset(torch.utils.data.Dataset):
#     def __init__(self, training_files, segment_size, n_fft, num_mels,
#                  hop_size, win_size, sampling_rate, fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
#                  device=None, fmax_loss=None, fine_tuning=False):
#         self.audio_files = training_files
#         random.seed(1234)
#         if shuffle:
#             random.shuffle(self.audio_files)
#         self.segment_size = segment_size
#         self.sampling_rate = sampling_rate
#         self.split = split
#         self.n_fft = n_fft
#         self.num_mels = num_mels
#         self.hop_size = hop_size
#         self.win_size = win_size
#         self.fmin = fmin
#         self.fmax = fmax
#         self.fmax_loss = fmax_loss
#         self.device = device
#
#     def __getitem__(self, index):
#         audio_filename = self.audio_files[index]
#         mel_filename = audio_filename.replace('wave', 'feats')
#
#         audio = np.load(audio_filename)
#         audio = torch.FloatTensor(audio)  # [1, T]
#         audio = audio.unsqueeze(0)
#
#
#         mel = np.load(mel_filename)
#         #mel = torch.from_numpy(mel)
#         mel = torch.from_numpy(mel).unsqueeze(0).transpose(-1, -2)
#
#         # if len(mel.shape) < 3:
#         #     mel = mel.unsqueeze(0)
#         # else:
#         #     pass
#         #
#
#
#         # frames_per_seg = math.ceil(self.segment_size / self.hop_size)
#
#         if len(mel.shape) < 3:
#             mel = mel.unsqueeze(0)
#
#         if self.split:
#             frames_per_seg = math.ceil(self.segment_size / self.hop_size)
#
#             if audio.size(1) >= self.segment_size:
#                 mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
#                 mel = mel[:, :, mel_start:mel_start + frames_per_seg]
#                 audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
#             else:
#                 mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
#                 audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
#
#         # if audio.size(1) >= self.segment_size:
#         #     mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
#         #     mel = mel[:, :, mel_start:mel_start + frames_per_seg]
#         #     audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
#         # else:
#         #     mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
#         #     audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
#         #
#         # mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
#         #                            self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
#         #                            center=False)
#         mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
#                                    self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
#                                    center=False)
#         return (mel.squeeze(), audio.squeeze(0), audio_filename, mel_loss.squeeze())
#
#     def __len__(self):
#         return len(self.audio_files)
#

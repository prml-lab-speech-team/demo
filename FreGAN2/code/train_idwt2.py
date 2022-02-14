import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
os.environ['KMP_WARNINGS'] = '0'
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp

from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist_custom
from fregan2_idwt2 import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, \
    generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

import librosa
from pypesq import pesq

torch.backends.cudnn.benchmark = True


def train(rank, a, h, n_gpus):
    '''
    :param rank: GPU number
    :param a: config
    :param h: hparams
    '''

    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=n_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator(h).to(device)
    msd = MultiScaleDiscriminator(h).to(device)

    # 1st GPU
    if rank == 0:

        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:  # Train from scratch

        state_dict_do = None
        last_epoch = -1
    else:  # Load
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank], find_unused_parameters=True).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank], find_unused_parameters=True).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist_custom(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device)

    # Distribute dataset into multi-gpu
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True, drop_last=True)

    if rank == 0:
        validset = MelDataset(sorted(validation_filelist), h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True) # [Batch , T], Full resolution
            y_mel = y_mel.to(device, non_blocking=True) # [Batch , T], Full resolution


            y = y.unsqueeze(1)  # [Batch, 1, T]

            # Generator
            y_g_hat = generator(x)

            # mel_spectrogram --> input.shape: [Batch, T]

            y_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()  # Update parameter

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss

            loss_mel24 = F.l1_loss(y_mel, y_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel24
            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error24 = F.l1_loss(y_mel, y_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Loss : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error24, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                             else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                             else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss", loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f, steps)
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error24, steps)

                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot24 = 0
                    val_pesq = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            # discriminator 통과안하고 mel loss만
                            x, y, _, y_mel = batch

                            y = y.to(device, non_blocking=True)
                            y_mel = y_mel.to(device, non_blocking=True)
                            x = x.to(device, non_blocking=True)
                            y_g_hat= generator(x)

                            # y_hat: 8192

                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                                     h.sampling_rate,
                                                                     h.hop_size,
                                                                     h.win_size, h.fmin, h.fmax_for_loss)


                            val_err_tot24 += F.l1_loss(y_mel, y_g_hat_mel).item()
                            if steps > 0 and j < 400:
                                audio_one_16000 = librosa.resample(y.squeeze().cpu().numpy(), 22050, 16000)
                                audio_two_16000 = librosa.resample(y_g_hat.squeeze().cpu().numpy(), 22050, 16000)
                                # PESQ
                                val_pesq += pesq(audio_one_16000, audio_two_16000, 16000)
                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0].cpu().numpy()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)


                        val_err24 = val_err_tot24 / (j + 1)
                        val_pesq = val_pesq / (400)
                        sw.add_scalar("validation/mel_spec_error", val_err24, steps)
                        sw.add_scalar("validation/pesq", val_pesq, steps)
                    generator.train()

            steps = steps + 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_training_file', default='/workspace/sh_lee/dataset/LJSpeech_22050/train')
    parser.add_argument('--input_validation_file', default='/workspace/sh_lee/dataset/LJSpeech_22050/val')

    parser.add_argument('--checkpoint_path', default='/workspace/sh_lee/log_fregan/FreGAN_multi_idwt_v1')
    parser.add_argument('--config', default='./config_fregan2_multi_idwt_v1.json')

    # parser.add_argument('--checkpoint_path', default='/workspace/sh_lee/log_fregan/FreGAN_multi_idwt_v2')
    # parser.add_argument('--config', default='./config_fregan2_multi_idwt_v2.json')

    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=50000, type=int)
    parser.add_argument('--summary_interval', default=500, type=int)
    parser.add_argument('--validation_interval', default=10000, type=int)
    parser.add_argument('--fine_tuning', default=True, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        n_gpus = torch.cuda.device_count()
        print(n_gpus)
        mp.spawn(train, nprocs=n_gpus, args=(a, h, n_gpus))
    else:
        n_gpus = torch.cuda.device_count()
        train(0, a, h, n_gpus)


if __name__ == '__main__':
    main()

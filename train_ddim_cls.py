import yaml
import random
import argparse
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import time

import torch
import torchaudio
from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler

from accelerate import Accelerator
from diffusers import DDIMScheduler

from modules.autoencoder import AutoencoderKL
from modules.mel import LogMelSpectrogram
from model.unet import DiffTSE
from utils import save_plot, save_audio
from dataset import TSEDataset

parser = argparse.ArgumentParser()

# data loading settings
parser.add_argument('--data-path', type=str, default='../data/fsd2018/')
parser.add_argument('--timbre-path', type=str, default=None)
parser.add_argument('--audio-length', type=int, default=10)
parser.add_argument('--use-timbre-feature', type=bool, default=False)
parser.add_argument('--mel-length', type=int, default=1024)
parser.add_argument('--val-num', type=int, default=8)

# pre-trained model path
parser.add_argument('--autoencoder-path', type=str, default='modules/first_stage.pt')
parser.add_argument('--scale-factor', type=float, default=1.0)

# training settings
parser.add_argument("--amp", type=str, default='fp16')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--save-every', type=int, default=1)

# model configs
parser.add_argument('--autoencoder-config', type=str, default='modules/vae.yaml')
parser.add_argument('--diffusion-config', type=str, default='model/DiffTSE_cls.yaml')

# optimization
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-6)
parser.add_argument("--adam-epsilon", type=float, default=1e-08)

# steps for DDIM training
parser.add_argument("--num-train-steps", type=int, default=1000)
parser.add_argument("--num-infer-steps", type=int, default=100)

# log and random seed
parser.add_argument('--random-seed', type=int, default=2023)
parser.add_argument('--log-step', type=int, default=100)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--save-dir', type=str, default='ckpt/')


args = parser.parse_args()

with open(args.autoencoder_config, 'r') as fp:
    args.vae_config = yaml.safe_load(fp)

with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)

if os.path.exists(args.log_dir + '/pic') is False:
    os.makedirs(args.log_dir + '/pic')

if os.path.exists(args.log_dir + '/audio') is False:
    os.makedirs(args.log_dir + '/audio')

if os.path.exists(args.save_dir) is False:
    os.makedirs(args.save_dir)

n = open(args.log_dir + 'ddim_cls_log.txt', mode='w')
n.write('diff tse log')
n.close()

@torch.no_grad()
def eval_ddim(autoencoder, unet, scheduler, eval_loader, epoch=0, global_step=0, ddim_steps=100, eta=1):
    # noise generator for eval
    generator = torch.Generator(device=accelerator.device).manual_seed(args.random_seed)
    scheduler.set_timesteps(ddim_steps)

    unet.eval()
    for step, (mixture, _, target, _, _, cls, _, file_id) in enumerate(eval_loader):
        # compress by vae
        mixture = autoencoder.mel2emb(logmel(mixture))*args.scale_factor
        # timbre = logmel(timbre)
        cls = cls.long()
        target = autoencoder.mel2emb(logmel(target))*args.scale_factor

        # init noise
        noise = torch.randn(target.shape, generator=generator, device=mixture.device)
        pred = noise

        for t in scheduler.timesteps:
            pred = scheduler.scale_model_input(pred, t)
            model_output = unet(x=pred, t=t, mixture=mixture, cls=cls,
                                timbre=None, timbre_feature=None)
            pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                                  eta=eta, generator=generator).prev_sample

        pred = pred / args.scale_factor
        pred = autoencoder.emb2mel(pred)
        wav = autoencoder.mel2wav(pred)

        save_plot(pred, f'{args.log_dir}/pic/{global_step}_pred_{file_id[0]}.png')
        save_audio(f'{args.log_dir}/audio/{global_step}_pred_{file_id[0]}', 16000, wav)

        if os.path.exists(f'{args.log_dir}/pic/{file_id[0]}.png') is False:
            target = autoencoder.emb2mel(target)
            target_wav = autoencoder.mel2wav(target)
            save_plot(target, f'{args.log_dir}/pic/{file_id[0]}.png')
            save_audio(f'{args.log_dir}/audio/{file_id[0]}', 16000, target_wav)

        if step+1 >= args.val_num:
            break


if __name__ == '__main__':
    # Fix the random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'

    train_set = TSEDataset(data_dir=args.data_path, subset='train', length=args.audio_length,
                           use_timbre_feature=args.use_timbre_feature, timbre_path=args.timbre_path)
    val_set = TSEDataset(data_dir=args.data_path, subset='val', length=args.audio_length,
                         use_timbre_feature=args.use_timbre_feature, timbre_path=args.timbre_path)
    test_set = TSEDataset(data_dir=args.data_path, subset='test', length=args.audio_length,
                          use_timbre_feature=args.use_timbre_feature, timbre_path=args.timbre_path)

    train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size)
    # use this load for check generated audio samples
    eval_loader = DataLoader(val_set, num_workers=1, batch_size=1)
    # use these two loaders for benchmarks
    val_loader = DataLoader(val_set, num_workers=args.num_workers, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, num_workers=args.num_workers, batch_size=args.batch_size)

    # use accelerator for multi-gpu training
    accelerator = Accelerator(mixed_precision=args.amp)

    logmel = LogMelSpectrogram(target_length=args.mel_length).to(accelerator.device)

    autoencoder = AutoencoderKL(**args.vae_config['params'])
    checkpoint = torch.load(args.autoencoder_path, map_location='cpu')
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()
    autoencoder.to(accelerator.device)

    unet = DiffTSE(args.diff_config['diffwrap']).to(accelerator.device)
    ckpt = torch.load('ckpt/99.pt', map_location='cpu')['model']
    unet.load_state_dict(ckpt)

    noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_steps)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(unet.parameters(),
                                  lr=args.learning_rate,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay,
                                  eps=args.adam_epsilon,
                                  )
    # scaler = GradScaler()
    # put to accelerator
    unet, autoencoder, optimizer, train_loader, eval_loader, val_loader, test_loader = accelerator.prepare(
       unet, autoencoder, optimizer, train_loader, eval_loader, val_loader, test_loader
    )

    global_step = 0
    losses = 0

    for epoch in range(args.epochs):
        # unet.train()
        # for step, batch in enumerate(tqdm(train_loader)):
        #     # compress by vae
        #     mixture, _, target, _, _, cls, _ = batch
        #
        #     mixture = autoencoder.mel2emb(logmel(mixture))*args.scale_factor
        #     # timbre = logmel(timbre)
        #     cls = cls.long()
        #     target = autoencoder.mel2emb(logmel(target))*args.scale_factor
        #
        #     # adding noise
        #     noise = torch.randn(target.shape).to(accelerator.device)
        #     timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (noise.shape[0],),
        #                               device=accelerator.device,).long()
        #     noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
        #
        #     # inference
        #     pred = unet(x=noisy_target, t=timesteps, mixture=mixture, cls=cls,
        #                 timbre=None, timbre_feature=None)
        #
        #     # backward
        #     loss = loss_func(pred, noise)
        #     accelerator.backward(loss)
        #     optimizer.step()
        #     optimizer.zero_grad()
        #
        #     global_step += 1
        #     losses += loss.item()
        #
        #     if global_step % args.log_step == 0:
        #         n = open(args.log_dir + 'ddim_cls_log.txt', mode='a')
        #         n.write(time.asctime(time.localtime(time.time())))
        #         n.write('\n')
        #         n.write('Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}\n'.format(
        #             epoch + 1, args.epochs, step+1, len(train_loader), losses / args.log_step))
        #         n.close()
        #         losses = 0.0
        #
        #     # if global_step % 500 == 0:
        eval_ddim(autoencoder, unet, noise_scheduler, eval_loader, epoch, global_step, args.num_infer_steps)

        # if (epoch + 1) % args.save_every == 0:
        #     accelerator.wait_for_everyone()
        #     unwrapped_unet = accelerator.unwrap_model(unet)
        #     accelerator.save({
        #         "model": unwrapped_unet.state_dict(),
        #     }, args.save_dir+str(epoch)+'.pt')

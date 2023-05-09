import yaml
import librosa
import torch
import torchaudio
import soundfile as sf

from modules.autoencoder import AutoencoderKL
from modules.mel import LogMelSpectrogram

from dataset import TSEDataset

device = 'cuda'

if __name__ == "__main__":
    with open('modules/vae.yaml', 'r') as fp:
        config = yaml.safe_load(fp)

    autoencoder = AutoencoderKL(**config['params'])
    checkpoint = torch.load('modules/first_stage.pt')
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()
    autoencoder.to(device)

    M = LogMelSpectrogram(sr=16000, frame_length=1024, hop_length=160, n_mel=64,
                          f_min=0, f_max=8000,
                          target_length=1024).to(device)

    train_set = TSEDataset(data_dir='data/fsd2018/', subset='train', length=10,
                           use_timbre_feature=True, timbre_path='data/fsd2018/timbre_mae')
    test = train_set[666]
    mixture, timbre, target, _, _, _, timbre_feature = test
    mixture = mixture.unsqueeze(0)
    timbre = timbre.unsqueeze(0)
    target = target.unsqueeze(0)

    x = torch.cat([mixture, target, timbre], 0)
    mel = M(x.to(device))

    # encoding process
    z = autoencoder.mel2emb(mel)
    # decoding process
    mel_re = autoencoder.emb2mel(z)
    # vocode
    a = autoencoder.mel2wav(mel_re)

    a = (a.cpu().numpy())
    sf.write('mixture.wav', a[0], samplerate=16000)
    sf.write('target.wav', a[1], samplerate=16000)
    sf.write('timbre.wav', a[2], samplerate=16000)

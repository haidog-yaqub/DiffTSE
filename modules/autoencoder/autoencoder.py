import torch
import torch.nn as nn
from .modules import Encoder, Decoder
from .distributions import DiagonalGaussianDistribution

from ..vocoder.utilities import get_vocoder, vocoder_infer


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        image_key="fbank",
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
    ):
        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.vocoder = get_vocoder(None, "cpu")
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.flag_first_run = False

    def encode(self, x):
        # x = self.time_shuffle_operation(x)
        # B * C * T * Mels
        x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        # B * C*8 * T/4 * Mels/4
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = self.freq_merge_subband(dec)
        return dec

    # def decode_to_waveform(self, dec):
    #     dec = dec.squeeze(1).permute(0, 2, 1)
    #     # B * Mels/4 * T/4
    #     wav_reconstruction = vocoder_infer(dec, self.vocoder)
    #     return wav_reconstruction

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        if self.flag_first_run:
            print("Latent size: ", z.size())
            self.flag_first_run = False

        dec = self.decode(z)

        return dec, posterior

    def freq_split_subband(self, fbank):
        if self.subband == 1 or self.image_key != "stft":
            return fbank

        bs, ch, tstep, fbins = fbank.size()

        assert fbank.size(-1) % self.subband == 0
        assert ch == 1

        return (
            fbank.squeeze(1)
            .reshape(bs, tstep, self.subband, fbins // self.subband)
            .permute(0, 2, 1, 3)
        )

    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1 or self.image_key != "stft":
            return subband_fbank
        assert subband_fbank.size(1) == self.subband  # Channel dimension
        bs, sub_ch, tstep, fbins = subband_fbank.size()
        return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1)

    @torch.no_grad()
    def mel2emb(self, mel):
        # B * Mels * T
        mel = mel.permute(0, 2, 1)
        mel = mel.unsqueeze(1)
        z = self.encode(mel).sample()
        return z

    @torch.no_grad()
    def emb2mel(self, z):
        mel = self.decode(z)
        mel = mel.squeeze(1).permute(0, 2, 1)
        return mel

    @torch.no_grad()
    def mel2wav(self, mel):
        wav = self.vocoder(mel)
        return wav.squeeze(1)
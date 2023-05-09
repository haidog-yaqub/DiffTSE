import os
import random
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset


# training "average voice" encoder
class TSEDataset(Dataset):
    def __init__(self, data_dir, subset, sr=16000, length=10, use_timbre_feature=False, timbre_path=None):
        self.subset = subset
        self.path = data_dir
        meta = pd.read_csv(data_dir + 'meta.csv')
        self.meta = meta[meta['subset'] == subset]
        self.sr = sr
        self.train_samples = sr * length
        self.use_timbre_feature = use_timbre_feature
        if self.use_timbre_feature:
            assert timbre_path is not None
        self.timbre_path = timbre_path
        # random.seed(random_seed)

    def get_data(self, subfolder, audio_id):
        audio_path = os.path.join(self.path, subfolder)

        mixture_id = audio_id.replace('.wav', '.wav')
        mixture = os.path.join(audio_path, mixture_id)
        mixture, sr = torchaudio.load(mixture)
        if sr != self.sr:
            raise Exception("please prepare resampled audio before training")

        target_id = audio_id.replace('.wav', '_lab.wav')
        target = os.path.join(audio_path, target_id)
        target, _ = torchaudio.load(target)

        timbre_id = audio_id.replace('.wav', '_re.wav')
        timbre = os.path.join(audio_path, timbre_id)
        timbre, _ = torchaudio.load(timbre)

        if self.use_timbre_feature:
            feature_id = audio_id.replace('.wav', '_re.pt')
            timbre_path = os.path.join(self.timbre_path, subfolder)
            timbre_feature = os.path.join(timbre_path, feature_id)
            timbre_feature = torch.load(timbre_feature, map_location='cpu')
        else:
            timbre_feature = torch.zeros(1, 768)

        return mixture, timbre, target, timbre_feature

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        file_id = row['file']
        mixture, timbre, target, timbre_feature = self.get_data(row['folder'], file_id)
        onset = row['onset']
        offset = row['offset']
        cls = row['cls']

        max_start = max(mixture.shape[-1] - self.train_samples, 0)
        start = random.choice(range(max_start)) if max_start > 0 else 0
        end = int(start + self.train_samples)

        mixture = mixture[:, start:end]
        timbre = timbre[:, :]
        target = target[:, start:end]

        onset = min(max(onset-start/self.sr, 0), self.train_samples/self.sr)
        offset = max(0, min(offset-start/self.sr, self.train_samples/self.sr))

        if self.subset == 'train':
            return mixture.squeeze(), timbre.squeeze(), target.squeeze(), onset, offset, cls, timbre_feature.squeeze()
        else:
            return mixture.squeeze(), timbre.squeeze(), target.squeeze(), onset, offset, cls, timbre_feature.squeeze(), file_id

        # item = {'mixture': mixture, 'timbre': timbre, 'target': target,
        #         'onset': onset, 'offset': offset, 'cls': cls, 'timbre_feature': timbre_feature}
        # return item

    def __len__(self):
        return len(self.meta)
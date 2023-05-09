# DiffTSE
Offical Implementation of Deep Learning Project: Target Sound Extraction with Diffusion Probabilistic Models

- dataset: data loaders
- modules: autoencoder and vocoder
- model: unet backbone
- preprocessing: extract features from pre-trained model
- train_ldm: train diffusion model on latent space (exp1)
- train_spec: train TSE on spectrogram (exp2)

Part of the code is borrowed from the following repos.
https://github.com/CompVis/stable-diffusion
https://github.com/haoheliu/AudioLDM
https://github.com/jik876/hifi-gan

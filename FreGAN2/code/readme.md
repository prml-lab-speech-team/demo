# Fre-GAN 2: Fast and Efficient Frequency-consistent Audio Synthesis
### Sang-Hoon Lee, Ji-Hoon Kim, Kang-Eun Lee, Seong-Whan Lee

This is repository for the official Fre-GAN 2 implementation.
[Demo page](https://prml-lab-speech-team.github.io/demo/FreGAN2/) for audio samples.

**Abstract :**
Although recent advances in neural vocoder have shown significant improvement, most of these models have a trade-off between audio quality and computational complexity. Since the large model has a limitation on the low-resource devices, a more efficient neural vocoder should synthesize high-quality audio for practical applicability. In this paper, we present Fre-GAN 2, a fast and efficient high-quality audio synthesis model. For fast synthesis, Fre-GAN 2 only synthesizes low and high-frequency parts of the audio, and we leverage the inverse discrete wavelet transform to reproduce the target-resolution audio in the generator. Additionally, we also introduce adversarial periodic feature distillation, which makes the model synthesize high-quality audio with only a small parameter. The experimental results show the superiority of Fre-GAN 2 in audio quality. Furthermore, Fre-GAN 2 has a 10.91×generation acceleration, and the parameters are compressed by 21.23×than Fre-GAN.


## Pre-requisites
 Python >= 3.6

For PESQ, 
```
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
```
For DWT and iDWT
```
$ git clone https://github.com/fbcotter/pytorch_wavelets
$ cd pytorch_wavelets
$ pip install .
```

## References
* We thank the author of [HiFi-GAN](https://github.com/jik876/hifi-gan) for their great repository and paper.
* We refered [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets) for the DWT and iDWT.
* We used [PESQ](https://github.com/vBaiCai/python-pesq) for objective evaluation.

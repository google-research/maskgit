# MaskGIT: Masked Generative Image Transformer
Official Jax Implementation of the CVPR 2022 Paper

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskgit-masked-generative-image-transformer/image-generation-on-imagenet-512x512)](https://paperswithcode.com/sota/image-generation-on-imagenet-512x512?p=maskgit-masked-generative-image-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskgit-masked-generative-image-transformer/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=maskgit-masked-generative-image-transformer)

[[Paper](https://arxiv.org/abs/2202.04200)] [[Project Page](https://masked-generative-image-transformer.github.io/)] [[Demo Colab](https://colab.research.google.com/github/google-research/maskgit/blob/main/MaskGIT_demo.ipynb)]

Try the web demo with Class-conditional Image Synthesis and Editing here [![Replicate](https://replicate.com/google-research/maskgit/badge)](https://replicate.com/google-research/maskgit)

![teaser](imgs/teaser.png)

## Summary
MaskGIT is a novel image synthesis paradigm using a bidirectional transformer decoder. During training, MaskGIT learns to predict randomly masked tokens by attending to tokens in all directions. At inference time, the model begins with generating all tokens of an image simultaneously, and then refines the image iteratively conditioned on the previous generation. 

## Running pretrained models

Class conditional Image Genration models:

| Dataset  | Resolution | Model | Link | FID |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ImageNet  | 256 x 256 | Tokenizer | [checkpoint](https://storage.googleapis.com/maskgit-public/checkpoints/tokenizer_imagenet256_checkpoint)| 2.28 (reconstruction) |
| ImageNet  | 512 x 512 | Tokenizer | [checkpoint](https://storage.googleapis.com/maskgit-public/checkpoints/tokenizer_imagenet512_checkpoint)| 1.97 (reconstruction) |
| ImageNet  | 256 x 256 | MaskGIT Transformer |[checkpoint](https://storage.googleapis.com/maskgit-public/checkpoints/maskgit_imagenet256_checkpoint)| 6.06 (generation) |
| ImageNet  | 512 x 512 | MaskGIT Transformer | [checkpoint](https://storage.googleapis.com/maskgit-public/checkpoints/maskgit_imagenet512_checkpoint) | 7.32 (generation) |

You can run these models for class-conditional image **generation** and **editing** in the [demo Colab](https://colab.research.google.com/github/google-research/maskgit/blob/main/MaskGIT_demo.ipynb).

![teaser](imgs/class-conditional-teaser-small.png)

## Training
[Coming Soon]


## BibTeX

```
@InProceedings{chang2022maskgit,
  title = {MaskGIT: Masked Generative Image Transformer},
  author={Huiwen Chang and Han Zhang and Lu Jiang and Ce Liu and William T. Freeman},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2022}
}
```

## Disclaimer

This is not an officially supported Google product.

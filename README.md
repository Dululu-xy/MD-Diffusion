# MD Diffusion

This is a repository for the paper "[Diffusion Models for Multidimensional Seismic Noise Attenuation and Super-Resolution](https://library.seg.org/doi/10.1190/geo2023-0676.1)" (GEOPHYSICS).

The frame and some code are from [Janspiry/Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement).

## Dataset
All the data used in this paper is from [JintaoLee-Roger/SeismicSuperResolution](https://github.com/JintaoLee-Roger/SeismicSuperResolution).

## Code
All the code is [here](https://github.com/Dululu-xy/MD-Diffusion) and is being continuously updated...

## Model Structure
![MD Diffusion](https://github.com/user-attachments/assets/718a8196-c694-4821-9b77-e11737545291)

## Results

![MD Diffusion synthetic data denoise and superresolution](https://github.com/user-attachments/assets/ddffeed6-6e66-4def-b56e-5f20ec5228fe)

## Pretrained Models
The pre-trained models can be accessed below. Please select the model you need for download.
| Steps                             | Platform                                                     | 
| --------------------------------- | ------------------------------------------------------------ |
| 200 | [Google Drive](https://drive.google.com)\|[Baidu Yun](https://pan.baidu.com) |  
| 400 | [Google Drive](https://drive.google.com)\|[Baidu Yun](https://pan.baidu.com) | 
| 800 | [Google Drive](https://drive.google.com)\|[Baidu Yun](https://pan.baidu.com) | 
| 1000   | [Google Drive](https://drive.google.com)\|[Baidu Yun](https://pan.baidu.com) |
| 1200   | [Google Drive](https://drive.google.com)\|[Baidu Yun](https://pan.baidu.com) |
| 1500   | [Google Drive](https://drive.google.com)\|[Baidu Yun](https://pan.baidu.com) |
| 2000   | [Google Drive](https://drive.google.com)\|[Baidu Yun](https://pan.baidu.com) |

## Dependencies
- python 3.7.0
- torch 1.12.1+cu116

## Environment
```python
pip install -r requirement.txt
```

## Citation
If you find this work useful in your research, please consider citing:

Plain Text
```
Yuan Xiao, Kewen Li, Yimin Dou, Wentao Li, Zhixuan Yang, and Xinyuan Zhu, (2024), "Diffusion models for multidimensional seismic noise attenuation and superresolution," GEOPHYSICS 89: V479-V492.
```

BibTex
```latex
@article{article,
author = {Xiao, Yuan and Li, Kewen and Dou, Yimin and Li, Wentao and Yang, Zhixuan and Zhu, Xinyuan},
year = {2024},
month = {07},
pages = {V479-V492},
title = {Diffusion Models for Multidimensional Seismic Noise Attenuation and Super-Resolution},
journal = {GEOPHYSICS},
doi = {10.1190/geo2023-0676.1}
}
```

## Acknowledgements

Our work is based on the following theoretical works:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)
- [Deep Learning for Simultaneous Seismic Image Super-Resolution and Denoising](https://ieeexplore.ieee.org/document/9364884)
- [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)
- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)

Furthermore, we are benefitting a lot from the following projects:

- https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
- https://github.com/JintaoLee-Roger/SeismicSuperResolution
- https://github.com/dhale/ipf
- https://github.com/bhushan23/BIG-GAN
- https://github.com/lmnt-com/wavegrad
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/hejingwenhejingwen/AdaFM

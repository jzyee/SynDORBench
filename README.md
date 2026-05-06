

# SynDORBench: Benchmarking LVLM Perception Against Human Visual Capability <!-- omit in toc -->
![Static Badge](https://img.shields.io/badge/Task-HumanDetection-red) ![Static Badge](https://img.shields.io/badge/Task-ActionRecognition-red)

![Static Badge](https://img.shields.io/badge/Dataset-SynDORBench-blue)


![Banner](imgs/SynDORBench%20banner.png)
Figure 1. Conceptual overview of SynDORBench. The benchmark systematically varies physical visibility factors(lighting, scene type, camera azimuth/elevation, action pose) and DORI-calibrated viewing distance, evaluating the models on two tasks: 1) binary human detection and 2) open-ended action recognition


# Table of Contents <!-- omit in toc -->

- [About SynDORBench](#about-syndorbench)
- [Installation](#installation)
- [Evaluation](#evaluation)
- [Licence and Usage](#licence-and-usage)
- [References and Citation](#references-and-citation)



# About SynDORBench

This repository contains the evaluation code for SynDORBench, a benchmark for studying LVLM perceptual robustness under DORI-calibrated, physically constrained camera imaging conditions. SynDORBench evaluates model performance across controlled variations in viewing distance, lighting, camera geometry, and action pose using rendered synthetic images.



# Installation

Please refer to [INSTALL.md](INSTALL.md)

# Evaluation

Please refer to [EVAL.md](EVAL.md)

# Licence and Usage

This evaluation code is released under CC BY-NC-SA 4.0 for non-commercial academic research and reviewer inspection.

The code is intended to evaluate SynDORBench rendered images, annotations, and metadata. It does not include SMPL-X model files, AMASS motion files, BABEL source annotations, SMPLitex assets, or other upstream generation assets.

Users are responsible for complying with the original licences and terms of use of any third-party resources used with this code. Where upstream terms are more restrictive than the code-level licence, the upstream terms prevail.

# References and Citation

If you use SynDORBench, please cite SynDORBench and the following upstream resources:
```
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: {3D} Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10975--10985},
  year = {2019}
}

@conference{AMASS:ICCV:2019,
  title = {{AMASS}: Archive of Motion Capture as Surface Shapes},
  author = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {International Conference on Computer Vision},
  pages = {5442--5451},
  month = oct,
  year = {2019},
  month_numeric = {10}
}

@inproceedings{BABEL:CVPR:2021,
  title = {{BABEL}: Bodies, Action and Behavior with English Labels},
  author = {Punnakkal, Abhinanda R. and Chandrasekaran, Arjun and Athanasiou, Nikos and Quiros-Ramirez, Alejandra and Black, Michael J.},
  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  pages = {722--731},
  month = jun,
  year = {2021},
  doi = {},
  month_numeric = {6}
}

@article{casas2023smplitex,
  title={Smplitex: A generative model and dataset for 3d human texture estimation from single image},
  author={Casas, Dan and Comino-Trinidad, Marc},
  journal={arXiv preprint arXiv:2309.01855},
  year={2023}
}
```

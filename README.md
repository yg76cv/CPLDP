# Conditional Prompt Learning via Degradation Perception for Underwater Image Enhancement

This repository contains an implementation of Conditional Prompt Learning via Degradation Perception (CPLDP) for underwater image enhancement. The method learns degradation-aware soft prompts and uses them to condition an underwater enhancement network.

## Overview

Underwater images are divided into five degradation categories:

- Green color cast
- Blue color cast
- Green-blue color cast
- Uneven illumination
- Haze

The training pipeline has two stages:

1. Conditional prompt learning trains five degradation prompts and one positive reference prompt with frozen CLIP encoders.
2. Prompt-guided enhancement freezes the prompt module, and conditions the U-Net enhancement network.

## Data Preparation

The training code expects paired underwater and reference images with identical filenames:

```text
data/
|-- raw/
|   |-- image_001.png
|   `-- ...
|-- reference/
|   |-- image_001.png
|   `-- ...
`-- splits/
    |-- green.txt
    |-- blue.txt
    |-- green-blue.txt
    |-- Dark.txt
    `-- haze.txt
```

Each split file contains one filename per line. The mapping is:

```text
green.txt       -> 0
blue.txt        -> 1
green-blue.txt  -> 2
Dark.txt        -> 3
haze.txt        -> 4
```

Download Labels and weights
The non-overlapping 890-image degradation annotation split is available from Baidu Netdisk:
File: CPLDP-label
Link: https://pan.baidu.com/s/1Vc7ohTtA1EHpj5vXrY5pwg?
Extraction code: xtrs



## Citation
If you find our work useful for your research, please consider citing the paper:
```bibtex
@inproceedings{yao2026conditional,
  title={Conditional Prompt Learning via Degradation Perception for Underwater Image Enhancement},
  author={Yao, Mingze and Jiang, Zhiying and Fu, Xianping and Wang, Huibing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={14},
  pages={11892--11900},
  year={2026}
}
```


## Contact
If you have any questions, please feel free to reach out at ymz0284@dlmu.edu.cn.

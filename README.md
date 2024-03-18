# MSTNet
This repo holds codes of the paper: Multi-scale temporal network for continuous sign language recognition. [[paper]](https://arxiv.org/pdf/2204.03864.pdf)

---
### Notice

- This project is implemented in Pytorch (1.11.0+cu113). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- This project runs in pycharm, so you need to install pycharm

- The SLR is the main function.

---
### Data Preparation

1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).

2. After finishing dataset download, extract it.

---
### Inference
|            | WER on Dev | WER on Test |
|  --------  | ---------- | ----------- |
|   MSTNet   | 20.3%      | 21.4%       |

---
### Citation

If you find this repo useful in your research works, please consider citing:


```latex
@article{zhu2022multi,
  title={Multi-scale temporal network for continuous sign language recognition},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2204.03864},
  year={2022}
}
```
---
### Relevant paper

Continuous Sign Language Recognition via Temporal Super-Resolution Network. [[paper]](https://arxiv.org/pdf/2207.00928.pdf)

```latex
@article{zhu2022continuous,
  title={Continuous Sign Language Recognition via Temporal Super-Resolution Network},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2207.00928},
  year={2022}
}
```

Temporal superimposed crossover module for effective continuous sign language. [[paper]](https://arxiv.org/pdf/2211.03387.pdf)
```latex
@article{zhu2022temporal,
  title={Temporal superimposed crossover module for effective continuous sign language},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2211.03387},
  year={2022}
}
```

Continuous sign language recognition based on cross-resolution knowledge distillation. [[paper]](https://arxiv.org/pdf/2303.06820.pdf)
```latex
@article{zhu2023continuous,
  title={Continuous sign language recognition based on cross-resolution knowledge distillation},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2303.06820},
  year={2023}
}
```

Continuous Sign Language Recognition Based on Motor attention mechanism and frame-level Self-distillation. [[paper]](https://arxiv.org/pdf/2402.19118.pdf)
```latex
@article{zhu2024continuous,
  title={Continuous Sign Language Recognition Based on Motor attention mechanism and frame-level Self-distillation},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2402.19118},
  year={2024}
}
```

---
### Acknowledge

- Thanks Yuecong Min, Aiming Hao et al for sharing the code \([`link`](https://github.com/ycmin95/VAC_CSLR)

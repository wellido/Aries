# Source Code of Aries
#### [Aries: Efficient Testing of Deep Neural Networks via Labeling-Free Accuracy Estimation](https://arxiv.org/abs/2207.10942)

## Requirements
    - Tensorflow 2.3
    - Keras 2.4.3


## project structure
```
├── baselines                     # code for baselines, CES, PACE, Meta-set, Prediction-based, and Random
├── models                        # how to use dataset and prepare models            
├── results   
    ├── figures_for_rq2           # all figures of RQ2
├── utils                         # Tiny-ImageNet generator
├── Aries.py                      # Aries source code
├── example.py                    # example of using Aries
```


## Others
- PACE code is from https://github.com/pace2019/pace 
- CES code is from https://github.com/Lizn-zn/DNNOpAcc
- Meta-set code is from https://github.com/Simon4Yan/Meta-set
- CIFAR10-C data: https://zenodo.org/record/2535967#.YfRacRNKhQI
- Tiny-ImageNet-C data: https://zenodo.org/record/2469796#.YnPl6hMzZhE


## Citation
If you use the code in your research, please cite:
```bibtex
    @article{hu2022efficient,
    title={Aries: Efficient Testing of Deep Neural Networks via Labeling-Free Accuracy Estimation},
    author={Hu, Qiang and Guo, Yuejun and Xie, Xiaofei and Cordy, Maxime and Ma, Lei and Papadakis, Mike and Traon, Yves Le},
    journal={arXiv preprint arXiv:2207.10942},
    year={2022}
}
```

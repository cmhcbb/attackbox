# Attackbox

## Get started
To test PGD attack on MNIST:
```
python test_attack.py --attack PGD --dataset MNIST --model_dir [your_model_dir] --epsilon 0.1 --test_batch_size 100
```

To test OPT-attack on CIFAR10:
```
python test_attack.py --attack OPT_attack --dataset CIFAR10 --model_dir [your_model_dir] --epsilon 0.01 --test_batch_size 1
```

To test Sign-OPT on ImageNet:
```
python test_attack.py --attack OPT_attack --dataset CIFAR10 --model_dir [your_model_dir] --epsilon 0.01 --test_batch_size 1
```
## Define model structure
```

```

## Attack options:
```
python test_attack.py --attack some_attack --model_dir your_model_dir --epsilon some_number --test_batch num_batches --test_batch_size batch_size --targeted [True,False] --random_start [True,False]
```

## Supported attack
* White-box attack
    * $\ell_2$ attack:
        * PGD
        * C&W: https://arxiv.org/abs/1608.04644
        * FGSM: https://arxiv.org/abs/1412.6572

    * $\ell_\infty$ attack:
        * PGD-CE w/o random start: https://arxiv.org/abs/1706.06083
        * PGD-C&W w/o random start

* Soft-label Black-box attack
    * $\ell_2$ attack:
        * ZOO: https://arxiv.org/abs/1708.03999
    * $\ell_\infty$ attack:
        * NES: https://arxiv.org/abs/1804.08598
        * Sign-SGD: https://openreview.net/pdf?id=BJe-DsC5Fm

* Hard-label Black-box attack
    * $\ell_2$ and $\ell_\infty$ attack:
        * OPT-attack: https://arxiv.org/abs/1807.04457
        * Sign-OPT: https://arxiv.org/abs/1909.10773
        * HSJA: https://arxiv.org/abs/1904.02144

## Supported dataset
MNIST, CIFAR10, ImageNet

## Black-box attack Benchmark
### MNIST
| Attacks       | 0.1 |0.2|0.3           |
| ------------- |:-------------:|:-----:|----:|
| ZOO           | right-aligned | $1600 |
| NES           | centered      |   $12 |
| OPT-attack    | are neat      |    $1 |
| Sign-OPT      | are neat      |    $1 |
| PGD           | | |

### CIFAR10
| Attacks       | 0.1 |0.2|0.3           |
| ------------- |:-------------:|:-----:|----:|
| ZOO           | right-aligned | $1600 |
| NES           | centered      |   $12 |
| OPT-attack    | are neat      |    $1 |
| Sign-OPT      | are neat      |    $1 |
| PGD           | | |


## Download defense models
1. Install RobustBench package
```
pip install git+https://github.com/RobustBench/robustbench
```

2. Defense models will be downloaded automatically while loading

(Please checkout https://github.com/RobustBench/robustbench or download_defense_models.py for the list of available defense models.)

## Update (Dec 23, 2020)
We have made some changes to the exp management of this project. You can now test signopt quickly on cifar10 and clean model using the following bash script:
```
cd exp_scripts
bash signopt-lf.sh
bash signopt-l2.sh
```

More options are available as well:
```
bash signopt-lf.sh --model [e.g. Sehwag2020Hydra] --gpu [e.g. 0, auto (select the gpu with lowest memory)] --seed [e.g. 2]
bash signopt-l2.sh --model [e.g. Wu2020Adversarial] --gpu [e.g. 0, auto (select the gpu with lowest memory)] --seed [e.g. 2]
```
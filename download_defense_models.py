import os
from robustbench.utils import load_model
from robustbench.data import load_cifar10


model_dir = './model/defense_models/'
linf_models = [
    "Wu2020Adversarial_extra",
    "Carmon2019Unlabeled",
    "HYDRA",
    "Wang2020Improving",
    "Wu2020Adversarial",
    "Hendrycks2019Using",
    "Pang2020Boosting",
    "Zhang2020Attacks",
    "Rice2020Overfitting",
    "Huang2020Self",
    "Zhang2019Theoretically",
    "Chen2020Adversarial",
    "Engstrom2019Robustness",
    "Zhang2019You",
    "Wong2020Fast",
    "Ding2020MMA",
]

l2_models = [
    "Wu2020Adversarial",
    "Augustin2020Adversarial",
    "Engstrom2019Robustness",
    "Rice2020Overfitting",
    "Rony2019Decoupling	",
    "Ding2020MMA",
]


for model in linf_models:
    model = load_model(model_name=model, model_dir=model_dir, norm='Linf')
for model in l2_models:
    model = load_model(model_name=model, model_dir=model_dir, norm='L2')
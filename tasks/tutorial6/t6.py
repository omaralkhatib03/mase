#!/usr/bin/env python
# coding: utf-8

# In[3]:


checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"


# In[4]:


from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)


# In[5]:


from pathlib import Path
import dill

with open(f'{Path.cwd()}/../tutorial5/t5_best_model.pkl', "rb") as f:
    base_model = dill.load(f)


# In[ ]:


from chop.tools import get_tokenized_dataset

dataset, tokenizer = get_tokenized_dataset(
    dataset = dataset_name,
    checkpoint= tokenizer_checkpoint,
    return_tokenizer=True
)


# In[ ]:


from copy import deepcopy
from chop.tools import get_trainer

model = deepcopy(base_model)

trainer = get_trainer(
    model = model,
    tokenized_dataset = dataset,
    tokenizer = tokenizer,
    evaluate_metric = "accuracy",
    num_train_epochs=1
)

trainer.train()

b_res = trainer.evaluate()

baseline = b_res['eval_accuracy']

print(baseline)


# In[ ]:


import torch
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
    LinearBinaryScaling,
    LinearBinaryResidualSign,
)

search_space = {
    "int_width":[i for i in range(1, 4)],
    "block_widths": [i for i in range(2, 5)],
    "frac_width": [i for i in range(3, 6)],
    "linear_layer_choices": [
        torch.nn.Linear,
        LinearInteger, # 1
        LinearMinifloatIEEE, # 2
        LinearMinifloatDenorm, #3
        LinearLog, #4
        LinearBinaryScaling, #5
        LinearBinary, #6
        LinearBlockFP, #7
        LinearBlockLog #8
    ],
}


# In[12]:


from chop.tools.utils import deepsetattr
from copy import deepcopy


def getFloatConfig(trial, kwargs, appendWidthToBias=False):
    b_width = getPower2OfParam(trial, "bias_width", "frac_width")
    b_exp   = trial.suggest_int("bias_exponent_width", 1, b_width)

    d_width = getPower2OfParam(trial, "data_in_width", "frac_width")
    d_exp   = trial.suggest_int("data_exponent_width", 1, d_width)

    w_width = getPower2OfParam(trial, "weight_width", "frac_width")
    w_exp   = trial.suggest_int("weight_exponent_width", 1, w_width)

    width = "_width" if appendWidthToBias else ""

    kwargs["config"] = {
        "bias_width": b_width, # Main param
        "bias_exponent_width": b_exp,
        f"bias_exponent_bias{width}": trial.suggest_int(f"bias_exponent_bias{width}", 1, b_exp),

        "data_in_width": d_width, # Main param
        "data_in_exponent_width": d_exp,
        f"data_in_exponent_bias{width}": trial.suggest_int(f"data_exponent_bias{width}", 1, d_exp),

        "weight_width": w_width, # Main param
        "weight_exponent_width": w_exp,
        f"weight_exponent_bias{width}": trial.suggest_int(f"weight_exponent_bias{width}", 1, w_exp),
    }

    return kwargs

def getPower2OfParam(trial, param:str, ss_name:str) -> int:
    idx = trial.suggest_int(param, 0, len(search_space[ss_name])-1)
    return 2**search_space[ss_name][idx]

def construct_model(trial):
    trial_model = deepcopy(base_model)

    for name,layer in trial_model.named_modules():
        if (not isinstance(layer, torch.nn.Linear)):
            continue

        new_layer_cls = trial.suggest_categorical(
            f"{name}_type",
            search_space["linear_layer_choices"]
        )

        if (new_layer_cls == torch.nn.Linear):
            continue

        kwargs = {
            "in_features":layer.in_features,
            "out_features":layer.out_features
        }

        kwargs["config"] = {}

        if (new_layer_cls == LinearInteger):
           kwargs["config"] = {
                    "data_in_width":        getPower2OfParam(trial, "data_in_width", "int_width"),
                    "data_in_frac_width":   getPower2OfParam(trial, "data_in_frac_width", "frac_width"),
                    "weight_width":         getPower2OfParam(trial, "weight_width", "int_width"),
                    "weight_frac_width":    getPower2OfParam(trial, "weight_frac_width", "frac_width"),
                    "bias_width":           getPower2OfParam(trial, "bias_width", "int_width"),
                    "bias_frac_width":      getPower2OfParam(trial, "bias_frac_width", "frac_width"),
                }
        elif new_layer_cls in [
            LinearMinifloatIEEE,
            LinearMinifloatDenorm,
            LinearLog,
            LinearBlockFP,
            LinearBlockLog,
            LinearBlockMinifloat
        ]:
            kwargs = getFloatConfig(trial, kwargs, appendWidthToBias=new_layer_cls in [LinearBlockLog, LinearBlockMinifloat])

            if new_layer_cls in [LinearBlockLog, LinearBlockMinifloat, LinearBlockFP]:
                for param in ["weight_block_size", "data_in_block_size", "bias_block_size"]:
                    dim = trial.suggest_int(f"dim_{param}", 1, 2)
                    kwargs["config"][param] = [0]*dim
                    for i in range(0, dim):
                        kwargs["config"][param][i-1] = getPower2OfParam(trial, param, "block_widths")
        elif new_layer_cls == LinearBinary:

            kwargs["config"]["weight_stochastic"] = trial.suggest_int("weight_stochastic", 0, 1)
            kwargs["config"]["weight_bipolar"] = True

        elif new_layer_cls == LinearBinaryScaling:
            kwargs["config"]["data_in_bipolar"] = True
            kwargs["config"]["bias_bipolar"] = True
            kwargs["config"]["weight_bipolar"] = True

            kwargs["config"]["data_in_stochastic"] = trial.suggest_int("data_in_stochastic", 0, 1)
            kwargs["config"]["bias_stochastic"] = trial.suggest_int("bias_stochastic", 0, 1)
            kwargs["config"]["weight_stochastic"] = trial.suggest_int("weight_stochastic", 0, 1)
            kwargs["config"]["binary_training"] = trial.suggest_int("weight_stochastic", 0, 1)
        else:
            raise RuntimeError("Not Implemented")

        assert len(kwargs["config"]) > 0

        new_layer = new_layer_cls(**kwargs)

        new_layer.weight.data = layer.weight.data

        deepsetattr(trial_model, name, new_layer)

    return trial_model


# In[13]:


from chop.tools import get_trainer
import random

def objective(trial):

    model = construct_model(trial)

    trainer = get_trainer(
        model = model,
        tokenized_dataset = dataset,
        tokenizer = tokenizer,
        evaluate_metric = "accuracy",
        num_train_epochs=1
    )

    trainer.train()
    eval_results = trainer.evaluate()
    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]


# In[16]:


from optuna.samplers import GridSampler, RandomSampler, TPESampler

sampler = TPESampler()


# In[ ]:


import optuna

study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler
)

study.optimize(
    objective,
    n_trials=1,
    timeout = 60 * 60 * 24
)


# In[20]:


from optuna import Study
from optuna.trial import FrozenTrial
import pandas as pd

def samplerTrial(sampler_name:str, trials:int, sampler, f=objective) -> None:

    data = [0]*trials

    def record_accuracy_callback(stud:Study, fzt:FrozenTrial):
        print(f"Trial: {fzt.number}, Accuracy: {fzt.value}")
        data[fzt.number] ={"n":fzt.number, "accuracy":fzt.value}

    if (sampler == None):
        raise RuntimeError("No Sampler Provided")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"bert-tiny-nas-{sampler_name}-study",
        sampler=sampler
    )

    study.optimize(
        f,
        n_trials=trials,
        timeout=60*60*24,
        callbacks=[record_accuracy_callback]
    )

    df = pd.DataFrame(data)
    df.to_csv(f"{Path.cwd()}/sampler_run_{sampler_name}.csv", index=False)


# In[ ]:


samplerTrial(sampler_name="tpes_layers", trials=100, sampler=sampler, f=objective)


# In[24]:


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np


def plot(df_name:str) -> None:
    df = pd.read_csv(f"{Path.cwd()}/{df_name}.csv")

    # plt.scatter(x=df['n'], y=df['accuracy'] * 100, marker='x', s=20)

    # plt.axhline(y=baseline * 100, linestyle='--', linewidth=0.8)
    # plt.legend(['Accuracy', 'Baseline'], bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.xlabel("Trials")
    # plt.ylabel("Accuracy %")
    # plt.title("Trials vs Accuracy")

    # plt.show()

    fig = plt.figure()

    plt.scatter(x=df['n'], y=df['accuracy'].cummax() * 100, marker='x', s=20)

    plt.axhline(y=baseline * 100, linestyle='--', linewidth=0.8)
    lgd = plt.legend(['Accuracy', 'Baseline'], bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel("Trials")
    plt.ylabel("Accuracy %")
    plt.title("Trials vs Accuracy")
    plt.savefig(f"{Path.cwd()}/cp_{df_name}",  bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


# In[23]:


# plot('sampler_run_random')
# plot('tps')


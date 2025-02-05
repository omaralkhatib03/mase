#!/usr/bin/env python
# coding: utf-8

# In[6]:


checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"


# In[2]:


from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)


# In[4]:


from pathlib import Path
import dill

with open(f'{Path.cwd()}/../tutorial5/t5_best_model.pkl', "rb") as f:
    base_model = dill.load(f)


# In[7]:


from chop.tools import get_tokenized_dataset

dataset, tokenizer = get_tokenized_dataset(
    dataset = dataset_name,
    checkpoint= tokenizer_checkpoint,
    return_tokenizer=True
)


# In[42]:


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
    "frac_width": [i for i in range(3, 6)],
    "linear_layer_choices": [
        torch.nn.Linear,
        LinearInteger,
    ],
}


# In[47]:


from chop.tools.utils import deepsetattr
from copy import deepcopy

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

        if (new_layer_cls == LinearInteger):
           kwargs["config"] = {
                    "data_in_width":        getPower2OfParam(trial, "data_in_width", "int_width"),
                    "data_in_frac_width":   getPower2OfParam(trial, "data_in_frac_width", "frac_width"),
                    "weight_width":         getPower2OfParam(trial, "weight_width", "int_width"),
                    "weight_frac_width":    getPower2OfParam(trial, "weight_frac_width", "frac_width"),
                    "bias_width":           getPower2OfParam(trial, "bias_width", "int_width"),
                    "bias_frac_width":      getPower2OfParam(trial, "bias_frac_width", "frac_width"),
                }
        else:
            raise RuntimeError("Not Implemented")


        new_layer = new_layer_cls(**kwargs)

        new_layer.weight.data = layer.weight.data

        deepsetattr(trial_model, name, new_layer)

    return trial_model


# In[44]:


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


# In[45]:


from optuna.samplers import GridSampler, RandomSampler, TPESampler

sampler = TPESampler()


# In[48]:


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


# In[53]:


from optuna import Study
from optuna.trial import FrozenTrial
import pandas as pd

def samplerTrial(sampler_name:str, trials:int, sampler, f=objective) -> None:

    data = [0]*trials

    def record_accuracy_callback(stud:Study, fzt:FrozenTrial):
        print(f"Trial: {fzt.number}, Accuracy: {fzt.value}")
        data[fzt.number-1] ={"n":fzt.number, "accuracy":fzt.value}

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


# In[54]:


samplerTrial(sampler_name="tpes", trials=100, sampler=sampler, f=objective)


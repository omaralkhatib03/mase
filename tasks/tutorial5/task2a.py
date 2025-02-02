#!/usr/bin/env python
# coding: utf-8

# In[2]:


from chop.tools import get_tokenized_dataset

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True
)


# In[3]:


from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools import get_trainer

config = AutoConfig.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_config(config)

trainer = get_trainer(
    model = model,
    tokenized_dataset = dataset,
    tokenizer = tokenizer,
    evaluate_metric = "accuracy",
    num_train_epochs = 1
)

trainer.train()

eval_results = trainer.evaluate()

baseline = eval_results["eval_accuracy"]
print(baseline)


# In[4]:


import torch.nn as nn
from chop.nn.modules import Identity

search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices" : ["linear", "identity"]
}


# In[55]:


from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr

track = set({})

def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)
    ss_template = search_space

    for param in [
        "num_layers",
        "num_heads",
        "hidden_size",
        "intermediate_size"
    ]:
        chosen_idex = trial.suggest_int(param, 0, len(ss_template[param]) - 1)
        setattr(config, param, ss_template[param][chosen_idex])

    trial_model = AutoModelForSequenceClassification.from_config(config)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            track.add(name)
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                ss_template["linear_layer_choices"],
            )

            if new_layer_cls == "linear":
                continue
            elif new_layer_cls == "identity":
                new_layer = Identity()
                deepsetattr(trial_model, name, new_layer)
            else:
                raise ValueError(f"Unkown layer type: {new_layer_cls}")

    return trial_model


# In[48]:


from chop.tools import get_trainer

def objective(trial):
    model = construct_model(trial)

    trainer = get_trainer(
        model = model,
        tokenized_dataset = dataset,
        tokenizer = tokenizer,
        evaluate_metric = "accuracy",
        num_train_epochs = 1
    )

    trainer.train()

    eval_results = trainer.evaluate()

    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]


# In[38]:


import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler

def getGridSearchSpace():
    grid_search_space = {
        "num_layers": [i for i in range(0, len(search_space['num_layers']))],
        "num_heads": [i for i in range(0, len(search_space['num_heads']))],
        "hidden_size": [i for i in range(0, len(search_space['hidden_size']))],
        "intermediate_size": [i for i in range(0, len(search_space['intermediate_size']))],
    }

    sampler = RandomSampler()

    study = optuna.create_study(
        direction="maximize",
        study_name="bert-tiny-nas-study",
        sampler=sampler
    )

    study.optimize(
        lambda trial: construct_model(trial=trial),
        n_trials = 100,
        timeout=60*60*24
    )

    for name in track :
        grid_search_space[f'{name}_type'] = ["linear", "identity"]

    return grid_search_space


# In[ ]:


sampler = GridSampler(search_space=getGridSearchSpace()) # The sampler to use below


# In[50]:


study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler
)

study.optimize(
    objective,
    n_trials = 1,
    timeout=60*60*24
)


# In[51]:


from pathlib import Path
import dill

model = study.best_trial.user_attrs["model"].cpu()

with open(f"{Path.home()}/mase/tasks/tutorial5/t5_best_model.pkl", "wb") as f:
    dill.dump(model, f)


# In[52]:


from chop.pipelines import CompressionPipeline
from chop import MaseGraph

mg = MaseGraph(model)
pipe = CompressionPipeline()

quantization_config = {
    "by": "type",
    "default": {
        "config": {
            "name": None,
        }
    },
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

pruning_config = {
    "weight": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
    "activation": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
}

mg, _ = pipe(
    mg,
    pass_args={
        "quantize_transform_pass": quantization_config,
        "prune_transform_pass": pruning_config,
    },
)


# In[53]:


import pandas as pd
from optuna import Study
from optuna.trial import FrozenTrial

def samplerTrial(sampler_name:str, trials:int, sampler) -> None:

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
        objective,
        n_trials=trials,
        timeout=60*60*24,
        callbacks=[record_accuracy_callback]
    )

    df = pd.DataFrame(data)
    df.to_csv(f"{Path.home()}/mase/tasks/tutorial5/sampler_run_{sampler_name}.csv", index=False)


# In[ ]:


samplerTrial(sampler_name="grid", trials=100, sampler=sampler)


# In[ ]:


# import matplotlib.pyplot as plt
# import pandas as pd
# from pathlib import Path

# df_r = pd.read_csv(f"{Path.home()}/mase/tasks/tutorial5/random.csv")
# df_t = pd.read_csv(f"{Path.home()}/mase/tasks/tutorial5/tpes.csv")
# fig_pretrain = plt.figure()

# plt.scatter(x=df_r['n'], y=df_r['accuracy'] * 100, marker='x', s=20)
# plt.scatter(x=df_t['n'], y=df_t['accuracy'] * 100, marker='x', s=20)
# plt.axhline(y=baseline * 100, linestyle='--', linewidth=0.8)
# plt.legend(['Random Sampler', 'TPES Sampler', 'Baseline'])
# plt.xlabel("Trials")
# plt.ylabel("Accuracy %")
# plt.title("Trials vs Accuracy")
# plt.savefig(f"{Path.home()}/mase/tasks/tutorial5/samplers")
# plt.show()


# In[ ]:


# def samplerTrial(sampler_name:str, trials:int, sampler) -> None:

#     data = [0]*trials

#     def record_accuracy_callback(stud:Study, fzt:FrozenTrial):
#         print(f"Trial: {fzt.number}, Accuracy: {fzt.value}")
#         data[fzt.number-1] ={"n":fzt.number, "accuracy":fzt.value}

#     if (sampler == None):
#         raise RuntimeError("No Sampler Provided")


#     study = optuna.create_study(
#         direction="maximize",
#         study_name=f"bert-tiny-nas-{sampler_name}-study",
#         sampler=sampler
#     )

#     study.optimize(
#         objective,
#         n_trials=trials,
#         timeout=60*60*24,
#         callbacks=[record_accuracy_callback]
#     )

#     df = pd.DataFrame(data)
#     df.to_csv(f"{Path.home()}/mase/tasks/tutorial5/sampler_run_{sampler_name}.csv", index=False)



#!/usr/bin/env python
# coding: utf-8

# In[1]:


tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"


# In[2]:


from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import chop.passes as passes


# In[3]:


from pathlib import  Path
from chop import MaseGraph

mg = MaseGraph.from_checkpoint(f"{Path.home()}/mase/tasks/tutorial3/best_model")


# In[4]:


from chop.tools import get_tokenized_dataset, get_trainer

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True
)

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)


eval_results=trainer.evaluate()
print(f"Base Model Acc: {eval_results['eval_accuracy']}")
print(eval_results)


# In[5]:


pc = {
    "weight": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local"
    },
    "activation": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local"
    }
}


# In[16]:


import pandas as pd

def search_prune(strat: str) -> None:
    mg = MaseGraph.from_checkpoint(f"{Path.home()}/mase/tasks/tutorial3/best_model")
    prune_pretrain = [0 for i in range(1, 10)]
    prune_ac = [0 for i in range(1, 10)]
    pc_template = pc



    for i in range(7, 10):
        print(f'iter: {i}')
        pc_template["weight"]["sparsity"] = float(i) / 10.0
        pc_template["activation"]["sparsity"] = float(i) / 10.0
        pc_template["activation"]["method"] = strat
        pc_template["weight"]["method"] = strat
        mg, _ = passes.prune_transform_pass(mg, pc_template)

        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
        )

        eval_results = trainer.evaluate()

        prune_pretrain[i-1] = eval_results['eval_accuracy']

        trainer.train()

        eval_results2 = trainer.evaluate()

        prune_ac[i-1] = eval_results2['eval_accuracy']

    data = []

    for i in range(1, 10):
        data.append({
            "sparsity" : float(i)/10.0,
            "pretrain": prune_pretrain[i-1],
            "posttrain": prune_ac[i-1]
        })

    df = pd.DataFrame(data)
    df.to_csv(f"{Path.home()}/mase/tasks/tutorial4/prune_dt_{strat.replace('-', '_')}_34.csv", index=False)


# In[17]:

# search_prune("l1-norm")
search_prune("random")


# In[11]:


#import matplotlib.pyplot as plt
#import pandas as pd
#
#df_r = pd.read_csv(f"{Path.home()}/mase/tasks/tutorial4/prune_dt_random.csv")
#df_l = pd.read_csv(f"{Path.home()}/mase/tasks/tutorial4/prune_dt_l1_norm.csv")
#fig_pretrain = plt.figure()
#
#plt.scatter(x=df_r['sparsity'], y=df_r['pretrain'] * 100, marker='x', s=20)
#plt.scatter(x=df_r['sparsity'], y=df_r['posttrain'] * 100, marker='x', s=20)
#plt.scatter(x=df_l['sparsity'], y=df_l['pretrain'] * 100, marker='x', s=20)
#plt.scatter(x=df_l['sparsity'], y=df_l['posttrain'] * 100, marker='x', s=20)
#plt.axhline(y=eval_results['eval_accuracy'] * 100, linestyle='--', linewidth=0.8)
#plt.legend(['Pruning (Random)', 'Post Re-training (Random)', 'Pruning (L1-Norm)', 'Post Re-training (L1-Norm)', 'Baseline'])
#plt.xlabel("Sparsity Ratio")
#plt.ylabel("Accuracy %")
#plt.title("Sparsity Ratio vs Accuracy")
#plt.savefig(f"{Path.home()}/mase/tasks/tutorial4/sparse_job")
# plt.show()


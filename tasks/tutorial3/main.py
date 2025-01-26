from transformers import AutoModelForSequenceClassification
from pathlib import  Path
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer

mg = MaseGraph.from_checkpoint(f"{Path.home()}/mase/tasks/tutorial2/tutorial_2_lora")
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

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

print(f"Evaluation Accuracy: {eval_results['eval_accuracy']}")
print(eval_results)

qc = {
    "by": "type",
    "default": {
        "config":
        {
            "name": None
        }
    },
    "linear":{
        "config":{
            "name":"integer",
            # Data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # Weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # Bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    }
}

mg, _ = passes.quantize_transform_pass(mg, pass_args=qc)


trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)

eval_results = trainer.evaluate()
print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")


# search loop (Maybe ? idk)

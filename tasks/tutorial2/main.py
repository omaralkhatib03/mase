from chop.tools import get_trainer
from chop.passes.module import report_trainable_parameters_analysis_pass
from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset

checkpoint = "DeepWokLab/bert-tiny"
tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"

mg = MaseGraph(
    model,
    hf_input_names=[
        "input_ids",
        "attention_mask",
        "labels",
    ],
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)

_, _ = report_trainable_parameters_analysis_pass(mg.model)


for param in mg.model.bert.embeddings.parameters():
    param.requires_grad = False

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)

from datasets import load_dataset
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs

dataset = load_dataset("D:\\CSN")

model_args = ClassificationArgs(
    sliding_window=True,
    labels_list=[0, 1],
    learning_rate=1e-5,
    train_batch_size=64,
    max_seq_length=200,
    num_train_epochs=8,
    optimizer="AdamW",
)
model = ClassificationModel(
    "roberta", "microsoft/codebert-base", args=model_args, use_cuda=True
)

train_data = []
validation_data = []

shuffled_data = dataset["train"].shuffle(seed=42)
shuffled_validation_data = dataset["validation"].shuffle(seed=42)

for index in range(1000):
    row = [
        shuffled_data._getitem(index)["func_documentation_string"]
        + "<CODESPLIT>"
        + shuffled_data._getitem(index)["func_code_string"],
        0,
    ]
    train_data.append(row)
    for i in range(index + 1, index + 8):
        row = [
            shuffled_data._getitem(index)["func_documentation_string"]
            + "<CODESPLIT>"
            + shuffled_data._getitem(i)["func_code_string"],
            1,
        ]
        train_data.append(row)


for index in range(0, 500):
    row = [
        shuffled_validation_data._getitem(index)["func_documentation_string"]
        + "<CODESPLIT>"
        + shuffled_validation_data._getitem(index)["func_code_string"],
        0,
    ]
    validation_data.append(row)
    for i in range(index + 1, index + 8):
        row = [
            shuffled_validation_data._getitem(index)["func_documentation_string"]
            + "<CODESPLIT>"
            + shuffled_validation_data._getitem(i)["func_code_string"],
            1,
        ]
        validation_data.append(row)


train_data = pd.DataFrame(train_data, columns=["text", "labels"])
validation_data = pd.DataFrame(validation_data, columns=["text", "labels"])


model.train_model(train_data, output_dir="./output", eval_df=validation_data)

model.eval_model(validation_data)

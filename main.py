from datasets import load_dataset
import ast
from train import *
from after_train import *




# Data Paths:
train_path = "./data/train.labeled"
val_path = "./data/val.labeled"
unlabeled_val_path = "./data/val.unlabeled"
comp_path = "./data/comp.unlabeled"

# Data Preprocessing:
preprocessing(train_path, "train")
preprocessing(val_path, "val")
preprocessing(comp_path, "comp")

# Pre Traind Model:
checkpoint = 't5-base'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=260, padding=True, truncation=True)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, max_length=260, )
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define the file names for the training data
train_data_files = {
    'train': 'train.csv',
}

# Load the dataset from the CSV file
train_load_datasets = load_dataset("csv", data_files=train_data_files)

# Map each example to a dictionary with "id" and "translation" keys
dataset = train_load_datasets.map(lambda example: {
    "id": example["id"],
    "translation": ast.literal_eval(example["translation"])
})

# Split the dataset into training and validation sets (20% validation)
dataset = dataset["train"].train_test_split(test_size=0.2)

# Apply the preprocess function to tokenize the text in the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Train the model :
train_model(tokenized_datasets)

# Load Trained Model :
model_name_checkpoint = "/content/drive/MyDrive/data/checkpoint-6000"
tokenizer_checkpoint = AutoTokenizer.from_pretrained(model_name_checkpoint, model_max_length=260, padding=True,
                                                     truncation=True)
model_checkpoint = AutoModelForSeq2SeqLM.from_pretrained(model_name_checkpoint, max_length=260, )

valdata = {
    'val': 'val.csv',
}
val_load_datasets = load_dataset("csv", data_files=valdata)
val_dataset = val_load_datasets .map(lambda example: {
    "id": example["id"],
    "translation": ast.literal_eval(example["translation"])
})

val_english_translation = tagger(val_dataset["val"],tokenizer_checkpoint,model_checkpoint)
val_german_original=german_data(unlabeled_val_path)
write_to_file_labeled(val_german_original, val_english_translation, "val_207364332")

compdata = {
    'comp': 'comp.csv',
}
comp_raw_datasets = load_dataset("csv", data_files=compdata)
comp_dataset = comp_raw_datasets .map(lambda example: {
    "translation": ast.literal_eval(example["translation"])
})

comp_english_translation = tagger(comp_dataset["val"],tokenizer_checkpoint,model_checkpoint,True)
comp_german_original=german_data(comp_path)
write_to_file_labeled(comp_german_original, comp_english_translation, "comp_207364332")


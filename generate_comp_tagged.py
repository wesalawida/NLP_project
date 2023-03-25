from datasets import load_dataset
import ast
from after_train import *

unlabeld_path = ""
preprocessing(unlabeld_path, "comp")


compdata = {
    'comp': 'comp.csv',
}
comp_raw_datasets = load_dataset("csv", data_files=compdata)
comp_dataset = comp_raw_datasets .map(lambda example: {
    "translation": ast.literal_eval(example["translation"])
})

# Load Trained Model :
model_name_checkpoint = "/content/drive/MyDrive/data/checkpoint-6000"
tokenizer_checkpoint = AutoTokenizer.from_pretrained(model_name_checkpoint, model_max_length=260, padding=True,
                                                     truncation=True)



model_checkpoint = AutoModelForSeq2SeqLM.from_pretrained(model_name_checkpoint, max_length=260, )
comp_english_translation = tagger(comp_dataset["val"],tokenizer_checkpoint,model_checkpoint,True)
comp_german_original=german_data(unlabeld_path)
write_to_file_labeled(comp_german_original, comp_english_translation, "comp_207364332")
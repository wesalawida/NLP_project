from preprocessing import *
from project_evaluate import *
import numpy as np
from transformers import  Seq2SeqTrainingArguments, Seq2SeqTrainer


def calculate_evaluation_metrics(eval_predictions):
    """
       Calculates evaluation metrics for sequence-to-sequence models.
       Args:
           eval_predictions: A tuple of predicted and ground truth tokenized sequences.
       Returns:
           A dictionary containing the BLEU score.
    """
    predictions, ground_truth = eval_predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    ground_truth = np.where(ground_truth != -100, ground_truth, tokenizer.pad_token_id)
    decoded_ground_truth = tokenizer.batch_decode(ground_truth, skip_special_tokens=True)
    result = compute_metrics(decoded_predictions, decoded_ground_truth)
    result = {"bleu": result}
    return result


def train_model(tokenized_datasets):
    """
        Trains a sequence-to-sequence model on tokenized datasets.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir="/content/drive/MyDrive/data",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        do_train=True,
        greater_is_better=True,
        save_strategy='epoch',
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=calculate_evaluation_metrics,
    )

    SAVED_PATH = '/content/drive/MyDrive/data/'
    trainer.train()
    trainer.save_model(SAVED_PATH)
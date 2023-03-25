from tqdm import *
from project_evaluate import *
from preprocessing import *
def write_to_file_labeled(german_paragraphs, english_paragraphs, filename):
    out_put_file = filename + ".labeled"
    with open(out_put_file, "w",encoding='utf-8') as f:
        for german, english in zip(german_paragraphs, english_paragraphs):
            f.write("German:\n")
            f.write("\n".join(german))
            f.write("\n")
            f.write("English:\n")
            english_sentences = english.split(". ")
            f.write("\n".join(english_sentences))
            f.write("\n\n")


def tagger(dataset, tokenizer, model, comp=False):
    tagged_translations, true_translations = [], []

    for i, example in tqdm(enumerate(dataset)):
        input_text = example["translation"]["de"]
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output_ids = model.generate(input_ids, max_length=400, num_beams=5)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        tagged_translations.append(output_text)
        if not comp:
            true_translations.append(example["translation"]["en"])
    if not comp:
        result = compute_metrics(tagged_translations, true_translations)
        print(result)
    return tagged_translations

def german_data(path):
    comp_dict = unlabeld_data(path)
    german_data = []
    for dict in comp_dict:
        german_data.append(dict["german"])
    return german_data

from transformers import AutoTokenizer
import pandas as pd
import spacy
import re
from transformers import AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,AutoTokenizer

checkpoint = 't5-base'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=260, padding=True, truncation=True)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,max_length=260,)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def unlabeld_data(file_path):
    """
    Reads a file containing unlabeled data and returns a list of dictionaries.
    Parameters:
    file_path (str): The path to the file containing the unlabeled data.

    Returns:
    list: A list of dictionaries, where each dictionary represents a single data point. Each dictionary contains the following keys:
        - 'german': A list of German sentences.
        - 'root': A list of English words representing the roots of the German sentences.
        - 'modifiers' (optional): A list of lists of English words representing the modifiers of the German sentences.
    """
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    res = []
    dictt = {}
    new_sen = []
    flag = False
    for line in lines:
        line = line.strip()
        if line == "German:":
            flag = True
            dictt = {}
            new_sen = []
            continue
        if flag:
            if "Roots in English:" in line:
                dictt["german"] = new_sen
                colon_index = line.index(":")
                words = line[colon_index + 2:].split(", ")
                dictt["root"] = words

            elif "Modifiers in English:" in line:
                word_sets = line.split(": ")[1].split("), (")
                # Loop through each set of words and extract the individual words
                modifiers = []
                for word_set in word_sets:
                    words = word_set.replace("(", "").replace(")", "").split(", ")
                    modifiers.append(words)

                dictt["modifiers"] = modifiers
                flag = False
                res.append(dictt)

            else:
                new_sen.append(line)
    return res


def add_labels_todict(res, en_val):
    """
    Adds English labels to a list of dictionaries.
    This function adds an 'english' key to each dictionary in the input list, containing the corresponding English label
     from the input list.

    Parameters:
    res (list): A list of dictionaries, where each dictionary represents a single data point.
    en_val (list): A list of English labels, where each label corresponds to a single data point in the input list.

    Returns:
    list: A list of dictionaries, where each dictionary represents a single data point. Each dictionary contains the
     following keys:
        - 'german': A list of German sentences.
        - 'root': A list of English words representing the roots of the German sentences.
        - 'modifiers' (optional): A list of lists of English words representing the modifiers of the German sentences.
        - 'english': A string representing the English label for the data point.
    """
    for i, curr_dict in enumerate(res):
        curr_dict["english"] = en_val[i]
    return res


def list_of_list_dataset(path):
    """
       Reads a text file containing a list of sentence pairs and returns two lists of lists: one for English sentences,
        and one for German sentences.

       Parameters:
       path (str): The path to the input file.

       Returns:
       tuple: Two lists of lists. The first list contains lists of English sentences, and the second list
        contains lists of German sentences.
       """
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    flag = False
    eng_parg = []
    ger_parg = []

    res_eng = []
    res_ger = []
    for line in lines:
        line = line.strip()
        if line == "German:":
            res_ger.append(eng_parg)
            ger_parg = []
            flag = True
        elif line == "English:":
            res_eng.append(ger_parg)
            eng_parg = []
            flag = False
        elif flag and line != "":
            eng_parg.append(line)
        elif not flag and line != "":
            ger_parg.append(line)
    return res_eng, res_ger


def train_pre(file_path):
    """
       This function reads in a file containing parallel German and English sentences,
       extracts the root and modifiers for each English sentence using the Spacy model,
       and returns a list of dictionaries where each dictionary represents a sentence pair
       with the German and English sentences and their respective roots and modifiers.

       Parameters:
           file_path (str): The path to the file containing parallel German and English sentences.

       Returns:
           list: A list of dictionaries where each dictionary represents a sentence pair
           with the German and English sentences and their respective roots and modifiers.
       """
    res_eng, res_ger = list_of_list_dataset(file_path)
    spacy_model = spacy.load('en_core_web_sm')
    regex = re.compile(r'\b\w+\b')
    to_function = []
    curr_dict = {}
    for i in range(len(res_ger)):
        root = []
        modifiers = []
        if len(res_ger[i]) == len(res_eng[i]):
            curr_dict = {}
            for index, j in enumerate(res_eng[i]):
                doc = spacy_model(j)
                for token in doc:
                    if token.dep_ == "ROOT":
                        root.append(token.text)
                        h = [str(child) for child in token.children if regex.match(str(child))]
                        modifiers.append(h)
                        break
            curr_dict["german"] = res_ger[i]
            curr_dict["english"] = res_eng[i]
            curr_dict["root"] = root
            curr_dict["modifiers"] = modifiers
            to_function.append(curr_dict)
        else:
            curr_sen = "".join(res_eng[i])
            doc = spacy_model(curr_sen)
            for token in doc:
                if token.dep_ == "ROOT":
                    root.append(token.text)
                    h = [str(child) for child in token.children if regex.match(str(child))]
                    modifiers.append(h)
                    break
            curr_dict["german"] = res_ger[i]
            curr_dict["english"] = res_eng[i]
            curr_dict["root"] = root
            curr_dict["modifiers"] = modifiers
            to_function.append(curr_dict)

    return to_function


def from_dict_to_sentance(res):
    eng_data = []
    ger_data = []
    curr_eng = []
    curr_ger = []
    for i in res:
        ger_sentences = i["german"]
        modifiers = i["modifiers"]
        roots = i["root"]
        eng_sentences = i["english"]

        if (len(ger_sentences) == len(modifiers)) and (len(ger_sentences) == len(roots)) and (
                len(ger_sentences) == len(eng_sentences)):
            for j, sen in enumerate(ger_sentences):
                paragraph = "sentence in german :" + sen +" ,root of sentence in English:" + str(roots[j]) + " ,modifiers of root in English:" + str(
                    ' '.join(modifiers[j]))+". "
                # eng_data.append(eng_sentences[j])
                # ger_data.append(paragraph)
                curr_ger.append(paragraph)
                curr_eng.append(eng_sentences[j])
            eng_data.append(" ".join(curr_eng))
            ger_data.append(" ".join(curr_ger))
            curr_eng = []
            curr_ger = []
            # out_sen = out_sen + paragraph
            # dd.append(out_sen)
        else:
            paragraph = "sentence in german :" + "".join(ger_sentences) + " ,root of sentence in English :" + ' '.join(roots) + " ,modifiers of root in English:" + str(
                modifiers)
            ger_data.append(paragraph)
            eng_data.append(" ".join(eng_sentences))
    return eng_data, ger_data


def comp_dict_to_sen(res):
    """
        Given a list of dictionaries containing information about German sentences and their English translations,
        returns two lists of sentences where each sentence in the first list corresponds to the corresponding sentence
        in the second list. Each sentence in the second list contains information about the root and modifiers of the
        corresponding German sentence.

        Returns:
        - eng_data: a list of strings representing the English sentences
        - ger_data: a list of strings representing the German sentences with information about the root and modifiers
        of each sentence
        """
    ger_data = []
    curr_ger = []
    for index,i in enumerate(res):
        ger_sentences = i["german"]
        modifiers = i["modifiers"]
        roots = i["root"]
        if (len(ger_sentences) == len(modifiers)) and (len(ger_sentences) == len(roots)):
            for j, sen in enumerate(ger_sentences):
                paragraph = "sentence in german :" + sen + " ,root of sentence in English:" + str(
                    roots[j]) + " ,modifiers of root in English:" + str(
                    ' '.join(modifiers[j]))+". "

                curr_ger.append(paragraph)
            ger_data.append(" ".join(curr_ger))
            curr_ger = []
        else:

            paragraph = "sentence in german :" + "".join(ger_sentences) + " ,root of sentence in English :" + ' '.join(
                roots) + " ,modifiers of root in English:" + str(modifiers)
            ger_data.append(paragraph)
    return ger_data


j = 0


def preprocessing(path, file_name):
    global j

    if file_name == "train":
        train_dict = train_pre(path)
        eng, de = from_dict_to_sentance(train_dict)
    elif file_name == "val":
        val_file_path = path
        en_val, de_val = list_of_list_dataset(val_file_path)
        val_path_unlabeld = val_file_path.replace("labeled", "unlabeled")
        val_dict = unlabeld_data(val_path_unlabeld)
        val_dict = add_labels_todict(val_dict, en_val)
        eng, de = from_dict_to_sentance(val_dict)
    elif file_name == "comp":
        comp_path_unlabeld = path
        comp_dict = unlabeld_data(comp_path_unlabeld)
        de = comp_dict_to_sen(comp_dict)
        eng = ["" for _ in range(len(de))]


    id = []
    translation = []

    for i, (sen1, sen2) in enumerate(zip(eng, de)):
        id.append(j)
        j = j + 1
        translation.append({'de': sen2,'en': sen1})
    df = {'id': id, 'translation': translation}
    df = pd.DataFrame(df)
    df.to_csv(file_name + '.csv', index=False)


max_length = 128


def preprocess_function(examples):
    inputs = [ex["de"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True, padding="max_length")
    return model_inputs

"""
Preprocessing functions to obtain the vocabularies for predicting literals
"""
import json
import re
import logging
from tqdm import tqdm
import stanza
from stanza import DownloadMethod

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from preprocessing.vocabulary import Vocabulary
from models.utils import get_plm_transformer

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("get-vocabulary")

VALUE_NUM_SYMBOL = '{value}'


def strip_nl(nl, tokenizer, remove_stopwords=False, lower=True):
    '''
    Return keywords of nl query
    '''
    nl_keywords = []
    nl = nl.strip()
    nl = nl.replace(";", " ; ").replace(",", " , ").replace("?", " ? ").replace("\t", " ")
    nl = nl.replace("(", " ( ").replace(")", " ) ").replace('"', '').replace(".", " ")

    str_1 = re.findall("\"[^\"]*\"", nl)
    str_2 = re.findall("\'[^\']*\'", nl)
    float_nums = re.findall("[-+]?\d*\.\d+", nl)

    values = str_1 + str_2 + float_nums
    for val in values:
        nl = nl.replace(val.strip(), VALUE_NUM_SYMBOL)

    raw_keywords = nl.strip().split()
    for tok in raw_keywords:
        if remove_stopwords and tok in stop_words:
            continue

        if "." in tok:
            to = tok.replace(".", " . ").split()
            to = [t.lower() for t in to if len(t) > 0]
            nl_keywords.extend(to)
        elif "'" in tok and tok[0] != "'" and tok[-1] != "'":
            to = tokenizer(tok).sentences[0]
            to = [t.text.lower() for t in to.tokens if len(t.text) > 0]
            nl_keywords.extend(to)
        elif len(tok) > 0:
            if lower:
                nl_keywords.append(tok.lower())
            else:
                nl_keywords.append(tok)

    return nl_keywords


def build_questions_vocab(input_file_path, remove_stopwords=False):
    input_file = json.load(open(input_file_path, "r"))
    stanza_nlp = stanza.Pipeline('en',
                                 processors='tokenize',
                                 logging_level='WARNING',
                                 download_method=DownloadMethod.REUSE_RESOURCES)

    vocab = Vocabulary(sentence_model=get_plm_transformer())

    for index, entry in enumerate(tqdm(input_file)):
        # Get the question and other related data
        nl_question = entry["question"]
        nl_question_strip = strip_nl(nl_question,
                                     tokenizer=stanza_nlp,
                                     remove_stopwords=remove_stopwords,
                                     lower=False)

        for token in nl_question_strip:
            vocab.add_word(token)

        vocab.add_word(entry["db_id"])

    return vocab


def build_schema_vocab(input_file_path, remove_stopwords=False):
    input_file = json.load(open(input_file_path, "r"))

    vocab = Vocabulary(sentence_model=get_plm_transformer())
    tables_vocab = Vocabulary(sentence_model=get_plm_transformer())
    columns_vocab = Vocabulary(sentence_model=get_plm_transformer())
    columns_vocab.add_word("*")

    col_name_to_original_mapping = dict()
    table_name_to_original_mapping = dict()

    for index, db in enumerate(tqdm(input_file)):
        for i, (column, column_names, column_type) in enumerate(zip(db['column_names_original'],
                                                            db['column_names'],
                                                            db['column_types'])):
            table_id, column_name_original = column
            table_id, column_name = column_names

            col_name_to_original_mapping[column_name] = column_name_original.lower()

            table_name = db['table_names'][table_id]
            table_name_original = db['table_names_original'][table_id]

            table_name_to_original_mapping[table_name] = table_name_original.lower()

            vocab.add_word(column_name_original.lower())
            vocab.add_word(table_name_original.lower())

            tables_vocab.add_word(table_name_original.lower())
            columns_vocab.add_word(column_name_original.lower())

    return vocab, tables_vocab, columns_vocab, col_name_to_original_mapping, table_name_to_original_mapping


def build_dataset_vocab(questions_file_path, tables_file_path):
    questions_vocab = build_questions_vocab(questions_file_path)
    schema_vocab, tables_vocab, columns_vocab, _, _ = build_schema_vocab(tables_file_path)
    return schema_vocab, tables_vocab, columns_vocab

import nltk
import abc
from abc import abstractmethod
from nltk import TreebankWordDetokenizer, TreebankWordTokenizer
from ir_datasets import Dataset as IRDataset

from typing import Iterable, List
import abc
from abc import abstractmethod

import pandas as pd
from nltk import TreebankWordDetokenizer, TreebankWordTokenizer
from ir_datasets import Dataset as IRDataset


class WordTokenizer:
    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def tokenize(self, text: str):
        pass

    @abstractmethod
    def detokenize(self, tokens: List[str]):  # follows nltk's detokenize signature
        pass


class WordTokenizer:
    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def detokenize(self, tokens):
        pass


class DatasetCleaner:
    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def __call__(self, document):
        pass

    def clean(self, document):
        return self.__call__(document)


class DatasetPreprocessor:
    def __init__(self, dataset: IRDataset, tokenizer: WordTokenizer, preprocessor: DatasetCleaner = None):
        self.dataset = dataset
class DatasetCleaner:
    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def __call__(self, document: dict):  # document is a dictionary with keys 'text' and 'docno'
        pass

    def clean(self, document: dict):
        return self.__call__(document)


class DatasetPreprocessor:
    def __init__(self, dataset: IRDataset, tokenizer: WordTokenizer, preprocessor: DatasetCleaner = None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        if (preprocessor is None) or (preprocessor == 'default'):
            self.preprocessor = TokenRemover(['../stopwords/stopwords-en.txt'], self.tokenizer)
        else:
            self.preprocessor = preprocessor

    def __iter__(self):
        for document in self.dataset.get_corpus_iter():
            yield self.preprocessor(document)


class TokenRemover(DatasetCleaner):
    def __init__(self, stopwords_files: Iterable[str], tokenizer: WordTokenizer, stopword_max_length: int = 512):
        self.tokenizer = tokenizer
        self.stopwords = set()
        self.max_len = stopword_max_length
        for file in stopwords_files:
            with open(file, 'r') as f:
                self.stopwords = self.stopwords | set(f.read().splitlines())

        self.stopwords = set([word for word in self.stopwords if len(word) <= self.max_len])

    def __call__(self, document: dict):
        doc_text = document['text']
        doc_tokenized = self.tokenizer.tokenize(doc_text)
        doc_cleaned = [tok for tok in doc_tokenized if tok not in self.stopwords]
        cleaned_text = self.tokenizer.detokenize(doc_cleaned)

        return {'text': cleaned_text, 'docno': document['docno']}


class DoNothingPreprocessor(DatasetCleaner):
    def __call__(self, document: dict):
        return document


class NLTKTokenizer(WordTokenizer):
    def __init__(self, tokenizer_type: str = 'treebank'):
        if tokenizer_type == 'default' or tokenizer_type == 'treebank':
            self.nltk_tokenizer = TreebankWordTokenizer()
            self.nltk_detokenizer = TreebankWordDetokenizer()

    def tokenize(self, text: str):
        return self.nltk_tokenizer.tokenize(text)

    def detokenize(self, tokens: List[str]):
        return self.nltk_detokenizer.detokenize(tokens)


class HFTokenizer(WordTokenizer):
    def __init__(self, tokenizer):  # tokenizer is a result of AutoTokenizer.from_pretrained()
        self.tokenizer = tokenizer

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: List[str]):
        return self.tokenizer.from_tokens_to_text(tokens)




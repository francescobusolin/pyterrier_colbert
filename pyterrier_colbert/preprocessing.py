class CorpusIteratorPreprocessor:

    def __init__(self, corpus_iterator, tokenizer, preprocessor=None):
        self.corpus_iterator = corpus_iterator
        self.tokenizer = tokenizer
        if (preprocessor is None) or (preprocessor == 'default'):
            self.preprocessor = TokenRemover(['../stopwords/stopwords-en.txt'], self.tokenizer)
        else:
            self.preprocessor = preprocessor

    def __iter__(self):
        for document in self.corpus_iterator:
            yield self.preprocessor(document)


class TokenRemover:
    def __init__(self, stopwords_files, tokenizer):
        self.tokenizer = tokenizer
        self.stopwords = set()
        for file in stopwords_files:
            with open(file, 'r') as f:
                self.stopwords = self.stopwords | set(f.read().splitlines())

    def __call__(self, document):
        doc_text = document['text']
        doc_tokenized = self.tokenizer.tokenize(doc_text)
        doc_cleaned = [tok for tok in doc_tokenized if tok not in self.stopwords]
        cleaned_text = self.tokenizer.convert_tokens_to_string(doc_cleaned)

        return {'text': cleaned_text, 'docno': document['docno']}

class CorpusIteratorPreprocessor:
    _default_stopwords = set([
        'the', 'a', 'an', 'in', 'on', 'at', 'as', 'of', 'by',
        " ' ", "''", "``", "(", ")", "[", "]", "{", "}", ",", ".",
        ":", ";", "!", "?", "&", "-", "--", "...", "''", "``", "..."
                                                               '"', '""', "''", "``", "(", ")", "[", "]", "{", "}", ",",
        ".", ":", ";", "!", "?", "&", "-", "--", "..."
                                                 '<', '>', '=', '+', '*', '@', '#', '$', '%', '^', '_', '`', '~', '\\',
        '/', '|',
        '.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '@', '#', '$', '%',
        '^', '&', '*', '+', '=', '_', '`', '~'
    ])

    def __init__(self, corpus_iterator, tokenizer, preprocessor=None):
        self.corpus_iterator = corpus_iterator
        self.tokenizer = tokenizer
        if preprocessor is None:
            self.preprocessor = self._default_preprocessor
        else:
            self.preprocessor = preprocessor

    def __iter__(self):
        for document in self.corpus_iterator:
            yield self.preprocessor(document)

    def _default_preprocessor(self, document):
        # document has form {'text': 'some text', 'docno': number}
        doc_text = document['text']
        doc_tokenized = self.tokenizer.tokenize(doc_text)
        doc_cleaned = [tok for tok in doc_tokenized if tok not in self._default_stopwords]
        cleaned_text = self.tokenizer.convert_tokens_to_string(doc_cleaned)

        return {'text': cleaned_text, 'docno': document['docno']}



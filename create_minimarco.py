import ir_datasets
import pyterrier as pt

pt.init()
from pyterrier_colbert import extract_docs_from_qrel

dataset = extract_docs_from_qrel(dataset=
                                 ir_datasets.load("msmarco-passage/trec-dl-2019/judged"),
                                 n=-1,
                                 notebook=False)
dataset.to_csv("msmarco-passage.tsv", sep='\t', index=False)

import ir_datasets
import pandas as pd
from ir_datasets import Dataset as IRDataset
from tqdm import tqdm
from tqdm import tqdm_notebook


def extract_docs_from_qrel(dataset: IRDataset, n: int = -1, notebook: bool = True) -> pd.DataFrame:
    rows = []
    used = set()
    found = 0
    progress = tqdm_notebook if notebook else tqdm
    docstore = dataset.docs_store()
    for qrel in progress(dataset.qrels_iter(), total=dataset.qrels_count(), desc="Extracting docs from qrels",
                         leave=False):
        if 0 < n <= found:
            break
        docid = qrel.doc_id
        doc = docstore.get(docid)
        entry = {'doc_id': docid, 'text': doc.text}
        if docid not in used:
            used.add(docid)
            rows.append(entry)
            found += 1
    return pd.DataFrame(rows)


if __name__ == '__main__':
    split = 'trec-dl-2019'
    ir_data = ir_datasets.load(f"msmarco-passage/{split}/judged")

    docs = extract_docs_from_qrel(dataset=ir_data, n=-1, notebook=False)
    queries = pd.DataFrame(ir_data.queries_iter())
    qrels = pd.DataFrame(ir_data.qrels_iter())
    qrels = qrels[['query_id', 'iteration', 'doc_id', 'relevance']]

    docs.to_csv(f"msmarco-passage-{split}-docs.tsv", sep='\t', index=False, header=False)
    queries.to_csv(f"msmarco-passage-{split}-queries.tsv", sep='\t', index=False, header=False)
    qrels.to_csv(f"msmarco-passage-{split}-qrels.tsv", sep='\t', index=False, header=False)

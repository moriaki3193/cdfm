# -*- coding: utf-8 -*-
"""Data structures used in this package.
"""
import os
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from operator import itemgetter
import numpy as np
from .config import DTYPE
from .types import DocumentID, QueryID, Vector, Label


@dataclass(frozen=True)
class Document:
    """A ranking target entity.
    """
    id: DocumentID
    vec: Vector
    label: Label

    def __eq__(self, that) -> bool:
        con = (self.id == that.id)
        con = con & (self.label == that.label)
        return con & np.array_equal(self.vec, that.vec)


@dataclass(frozen=True)
class Query:
    """A query that contains documents.
    """
    id: QueryID
    docs: List[Document]

    def __eq__(self, that) -> bool:
        for d1, d2 in zip(self.docs, that.docs):
            if not d1 == d2:
                return False
        return self.id == that.id

    def extract_others(self, id_: DocumentID) -> List[DocumentID]:
        """Extract a list of document ids other than the given document.
        """
        return [doc.id for doc in self.docs if doc.id != id_]


@dataclass()
class Data:
    """A dataset that contains queries.
    """
    queries: List[Query]
    map: Dict[DocumentID, int] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        lol = [[doc.id for doc in query.docs] for query in self.queries]
        uniques = set(itertools.chain.from_iterable(lol))
        self.map = {doc_id: idx for idx, doc_id in enumerate(uniques)}

    @classmethod
    def from_file(cls, path: str, ndim: int,
                  tokenizer: str = ' ', splitter: str = ':',
                  comment_symb: str = '#', zero_indexed: bool = False):
        """Read data from the specified path.

        Parameters:
            path: path to a file.
            tokenizer: symbol for decomposing a into key-value pair.
            splitter: symbol for splitting a key-value pair.
            comment_symb: symbol representing the begining of inline comments.
            zero_indexed: whether the feature vectors 0-indexed or not.

        Line format:
            label qid:123 eid:abc 1:1.5 2:0.2 4:-0.9 ... # comment

        Returns:
            data: DataFrame object of the loaded dataset.
        """
        # Validation on a given path.
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} not found.')

        _extract_first = itemgetter(0)

        # Line parser
        def _parse(line: str) -> Tuple[QueryID, Document]:
            line = line.rstrip()
            elems = _extract_first(line.split(comment_symb))
            tokens = elems.rstrip().split(tokenizer)
            label = DTYPE(tokens[0])
            _, qid = tokens[1].split(splitter)
            _, id_ = tokens[2].split(splitter)
            vec = np.zeros(ndim, dtype=DTYPE)
            for kv in tokens[3:]:
                nth_dim, raw_val = kv.split(splitter)
                dim_idx = int(nth_dim) if zero_indexed else int(nth_dim) - 1
                vec[dim_idx] = DTYPE(raw_val)
            return qid, Document(id_, vec, label)

        # Parse lines.
        query_ids: List[QueryID] = []
        documents: List[Document] = []
        with open(path, mode='r') as fp:
            line = fp.readline()
            while line:
                qid, doc = _parse(line)
                query_ids.append(qid)
                documents.append(doc)
                line = fp.readline()

        # Group documents into a query.
        q2i = {qid: idx for idx, qid in enumerate(set(query_ids))}
        i2q = {idx: qid for qid, idx in q2i.items()}
        doc_ls: List[List[Document]] = [[] for _ in range(len(q2i))]
        for qid, doc in zip(query_ids, documents):
            doc_ls[q2i[qid]].append(doc)

        queries = [Query(i2q[idx], docs) for idx, docs in enumerate(doc_ls)]
        return cls(queries)

    def lookup(self, id_: DocumentID) -> int:
        """Lookup an index of the document.

        If there is no registered document id, return `max(Index) + 1`.
        """
        return self.map.get(id_, len(self.map))
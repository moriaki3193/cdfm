# -*- coding: utf-8 -*-
"""Data structures used in this package.
"""
import itertools
from dataclasses import dataclass, field
from typing import Dict, List
from .types import DocumentID, QueryID, Vector


@dataclass(frozen=True)
class Document:
    """A ranking target entity.
    """
    id: DocumentID
    vec: Vector


@dataclass(frozen=True)
class Query:
    """A query that contains documents.
    """
    id: QueryID
    docs: List[Document]

    def extract_others(self, id_: DocumentID) -> List[DocumentID]:
        """Extract a list of document ids other than the given document.
        """
        return [doc.id for doc in self.docs if doc.id != id_]


@dataclass()
class Data:
    """A dataset that contains queries.
    """
    queries: List[Query]
    map: Dict[DocumentID, int] = field(init=False)

    def __post_init__(self) -> None:
        lol = [[doc.id for doc in query.docs] for query in self.queries]
        uniques = set(itertools.chain.from_iterable(lol))
        self.map = {doc_id: idx for idx, doc_id in enumerate(uniques)}

    def lookup(self, id_: DocumentID) -> int:
        """Lookup an index of the document.

        If there is no registered document id, return `max(Index) + 1`.
        """
        return self.map.get(id_, len(self.map))

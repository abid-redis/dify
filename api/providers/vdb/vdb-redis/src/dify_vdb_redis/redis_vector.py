"""Redis vector store backed by Redis search (Redis 8+, Redis Stack, Redis Cloud/Software)."""

import functools
import json
import re
import uuid
from array import array
from typing import Any, override

import redis
from pydantic import BaseModel, model_validator
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from configs import dify_config
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from models.dataset import Dataset

# Metadata keys Dify filters or deletes on; each is indexed as a TAG field.
TAG_FIELDS = ("document_id", "app_id", "annotation_id")
_DELETE_BATCH_SIZE = 1000


class RedisVectorConfig(BaseModel):
    url: str

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values: dict[str, Any]):
        if not values.get("url"):
            raise ValueError("config REDIS_VECTOR_URL is required")
        return values


@functools.cache
def _get_client(url: str) -> redis.Redis:
    """One connection pool per URL, shared by the RedisVector instances Dify creates per request."""
    return redis.Redis.from_url(url, decode_responses=True)


def _escape_tag(value: str) -> str:
    return re.sub(r"([^A-Za-z0-9_])", r"\\\1", value)


def _is_missing_index(error: ResponseError) -> bool:
    message = str(error).lower()
    return "no such index" in message or "unknown index name" in message


class RedisVector(BaseVector):
    """Stores each chunk as a HASH `{collection}:{doc_id}` indexed by Redis search."""

    def __init__(self, collection_name: str, config: RedisVectorConfig):
        super().__init__(collection_name)
        self._client = _get_client(config.url)
        self._index = self._client.ft(collection_name)
        self._key_prefix = f"{collection_name}:"

    @override
    def get_type(self) -> str:
        return VectorType.REDIS

    @override
    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        self._create_index(len(embeddings[0]))
        return self.add_texts(texts, embeddings)

    @override
    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        ids = []
        pipe = self._client.pipeline(transaction=False)
        for doc, embedding in zip(documents, embeddings):
            metadata = doc.metadata or {}
            doc_id = metadata.get("doc_id") or str(uuid.uuid4())
            mapping: dict[str | bytes, str | bytes] = {
                "page_content": doc.page_content,
                "metadata": json.dumps(metadata),
                "embedding": array("f", embedding).tobytes(),
            }
            mapping.update({field: str(metadata[field]) for field in TAG_FIELDS if metadata.get(field)})
            pipe.hset(self._key(doc_id), mapping=mapping)
            ids.append(doc_id)
        pipe.execute()
        return ids

    @override
    def text_exists(self, id: str) -> bool:
        return bool(self._client.exists(self._key(id)))

    @override
    def delete_by_ids(self, ids: list[str]):
        if ids:
            self._client.delete(*(self._key(id) for id in ids))

    @override
    def delete_by_metadata_field(self, key: str, value: str):
        if key not in TAG_FIELDS:
            raise ValueError(f"Redis vector store cannot filter on metadata field '{key}'")
        query = Query(f"@{key}:{{{_escape_tag(value)}}}").no_content().paging(0, _DELETE_BATCH_SIZE).dialect(2)
        while keys := [doc.id for doc in self._search(query)]:
            self._client.delete(*keys)

    @override
    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        top_k = self._top_k(kwargs, default=4)
        query = (
            Query(f"({self._document_filter(kwargs) or '*'})=>[KNN {top_k} @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("page_content", "metadata", "distance")
            .paging(0, top_k)
            .dialect(2)
        )
        score_threshold = float(kwargs.get("score_threshold") or 0.0)
        docs = []
        for result in self._search(query, {"vec": array("f", query_vector).tobytes()}):
            score = 1 - float(result.distance)
            if score >= score_threshold:
                docs.append(self._to_document(result, score))
        return docs

    @override
    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        top_k = self._top_k(kwargs, default=5)
        terms = re.findall(r"\w+", query)
        if not terms:
            return []
        search_query = (
            Query(f"@page_content:({'|'.join(terms)}) {self._document_filter(kwargs)}".strip())
            .with_scores()
            .return_fields("page_content", "metadata")
            .paging(0, top_k)
            .dialect(2)
        )
        return [self._to_document(result, float(result.score)) for result in self._search(search_query)]

    @override
    def delete(self):
        try:
            self._index.dropindex(delete_documents=True)
        except ResponseError as e:
            if not _is_missing_index(e):
                raise

    def _create_index(self, dimension: int):
        # FT.CREATE is atomic, so concurrent creators need no lock: the losers get "Index already exists".
        fields = [
            TextField("page_content"),
            *(TagField(field) for field in TAG_FIELDS),
            VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": dimension, "DISTANCE_METRIC": "COSINE"}),
        ]
        try:
            self._index.create_index(
                fields, definition=IndexDefinition(prefix=[self._key_prefix], index_type=IndexType.HASH)
            )
        except ResponseError as e:
            if "index already exists" not in str(e).lower():
                raise

    def _search(self, query: Query, params: dict[str, Any] | None = None) -> list[Any]:
        try:
            return self._index.search(query, query_params=params).docs
        except ResponseError as e:
            if _is_missing_index(e):
                return []
            raise

    def _key(self, doc_id: str) -> str:
        return f"{self._key_prefix}{doc_id}"

    @staticmethod
    def _top_k(kwargs: dict[str, Any], default: int) -> int:
        top_k = kwargs.get("top_k", default)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k

    @staticmethod
    def _document_filter(kwargs: dict[str, Any]) -> str:
        document_ids = kwargs.get("document_ids_filter")
        if not document_ids:
            return ""
        return f"@document_id:{{{'|'.join(_escape_tag(str(id)) for id in document_ids)}}}"

    @staticmethod
    def _to_document(result: Any, score: float) -> Document:
        metadata = json.loads(result.metadata)
        metadata["score"] = score
        return Document(page_content=result.page_content, metadata=metadata)


class RedisVectorFactory(AbstractVectorFactory):
    @override
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> RedisVector:
        if dataset.index_struct_dict:
            collection_name: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
        else:
            collection_name = Dataset.gen_collection_name_by_id(dataset.id)
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VectorType.REDIS, collection_name))

        return RedisVector(
            collection_name=collection_name,
            config=RedisVectorConfig(url=dify_config.REDIS_VECTOR_URL or ""),
        )

import json
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import dify_vdb_redis.redis_vector as redis_vector_module
import pytest
from dify_vdb_redis.redis_vector import RedisVector, RedisVectorConfig, RedisVectorFactory
from redis.exceptions import ResponseError

from core.rag.models.document import Document

COLLECTION = "Vector_index_abc_Node"


@pytest.fixture
def from_url(monkeypatch):
    from_url = MagicMock()
    monkeypatch.setattr(redis_vector_module.redis.Redis, "from_url", from_url)
    redis_vector_module._get_client.cache_clear()
    yield from_url
    redis_vector_module._get_client.cache_clear()


@pytest.fixture
def client(from_url):
    return from_url.return_value


@pytest.fixture
def vector(client):
    return RedisVector(COLLECTION, RedisVectorConfig(url="redis://localhost:6379"))


def _hit(metadata, **fields):
    return SimpleNamespace(id=f"{COLLECTION}:{metadata['doc_id']}", metadata=json.dumps(metadata), **fields)


def _query_string(index):
    return index.search.call_args.args[0].query_string()


def test_config_requires_url():
    with pytest.raises(ValueError, match="REDIS_VECTOR_URL is required"):
        RedisVectorConfig(url="")


def test_init_uses_url_and_collection_index(from_url, client, vector):
    from_url.assert_called_once_with("redis://localhost:6379", decode_responses=True)
    client.ft.assert_called_once_with(COLLECTION)
    assert vector.get_type() == "redis"


def test_instances_share_one_client_per_url(from_url, vector):
    RedisVector("Other_Node", RedisVectorConfig(url="redis://localhost:6379"))
    from_url.assert_called_once()


def test_create_builds_index_then_adds_texts(client, vector):
    docs = [Document(page_content="hello", metadata={"doc_id": "d1", "document_id": "doc-1"})]

    assert vector.create(docs, [[0.1, 0.2, 0.3]]) == ["d1"]

    fields, kwargs = client.ft.return_value.create_index.call_args
    names = [field.name for field in fields[0]]
    assert names == ["page_content", "document_id", "app_id", "annotation_id", "embedding"]
    assert "DIM" in fields[0][-1].args
    assert 3 in fields[0][-1].args
    assert f"{COLLECTION}:" in kwargs["definition"].args
    client.pipeline.return_value.hset.assert_called_once()


def test_create_index_tolerates_existing_index_only(client, vector):
    client.ft.return_value.create_index.side_effect = ResponseError("Index already exists")
    vector._create_index(3)

    client.ft.return_value.create_index.side_effect = ResponseError("boom")
    with pytest.raises(ResponseError):
        vector._create_index(3)


def test_add_texts_writes_hash_per_document(client, vector):
    metadata = {"doc_id": "d1", "document_id": "doc-1", "annotation_id": "a1", "dataset_id": "ds"}
    ids = vector.add_texts([Document(page_content="hello", metadata=metadata)], [[1.0, 2.0]])

    assert ids == ["d1"]
    pipe = client.pipeline.return_value
    key = pipe.hset.call_args.args[0]
    mapping = pipe.hset.call_args.kwargs["mapping"]
    assert key == f"{COLLECTION}:d1"
    assert mapping == {
        "page_content": "hello",
        "metadata": json.dumps(metadata),
        "embedding": array("f", [1.0, 2.0]).tobytes(),
        "document_id": "doc-1",
        "annotation_id": "a1",
    }
    pipe.execute.assert_called_once()


def test_text_exists_and_delete_by_ids(client, vector):
    client.exists.return_value = 1
    assert vector.text_exists("d1") is True
    client.exists.assert_called_once_with(f"{COLLECTION}:d1")

    vector.delete_by_ids([])
    client.delete.assert_not_called()
    vector.delete_by_ids(["d1", "d2"])
    client.delete.assert_called_once_with(f"{COLLECTION}:d1", f"{COLLECTION}:d2")


def test_search_by_vector_filters_and_applies_threshold(client, vector):
    index = client.ft.return_value
    index.search.return_value.docs = [
        _hit({"doc_id": "d1"}, page_content="near", distance="0.1"),
        _hit({"doc_id": "d2"}, page_content="far", distance="0.8"),
    ]

    docs = vector.search_by_vector([1.0, 2.0], top_k=2, score_threshold=0.5, document_ids_filter=["a-1", "b-2"])

    assert _query_string(index) == r"(@document_id:{a\-1|b\-2})=>[KNN 2 @embedding $vec AS distance]"
    assert index.search.call_args.kwargs["query_params"] == {"vec": array("f", [1.0, 2.0]).tobytes()}
    assert [doc.page_content for doc in docs] == ["near"]
    assert docs[0].metadata == {"doc_id": "d1", "score": pytest.approx(0.9)}


def test_search_by_vector_without_filter_matches_all(client, vector):
    client.ft.return_value.search.return_value.docs = []
    assert vector.search_by_vector([1.0]) == []
    assert _query_string(client.ft.return_value) == "(*)=>[KNN 4 @embedding $vec AS distance]"


def test_search_rejects_invalid_top_k(vector):
    with pytest.raises(ValueError, match="top_k"):
        vector.search_by_vector([1.0], top_k=0)


def test_search_by_full_text_tokenizes_query(client, vector):
    index = client.ft.return_value
    index.search.return_value.docs = [_hit({"doc_id": "d1"}, page_content="redis search", score="1.5")]

    docs = vector.search_by_full_text("redis: search!", document_ids_filter=["doc-1"])

    assert _query_string(index) == r"@page_content:(redis|search) @document_id:{doc\-1}"
    assert docs[0].metadata == {"doc_id": "d1", "score": 1.5}


def test_search_by_full_text_without_terms_skips_query(client, vector):
    assert vector.search_by_full_text("?! ") == []
    client.ft.return_value.search.assert_not_called()


def test_search_returns_empty_when_index_missing(client, vector):
    client.ft.return_value.search.side_effect = ResponseError("Vector_index_abc_Node: no such index")
    assert vector.search_by_vector([1.0]) == []
    assert vector.search_by_full_text("hello") == []


def test_delete_by_metadata_field_deletes_matching_keys(client, vector):
    index = client.ft.return_value
    batch = SimpleNamespace(docs=[SimpleNamespace(id=f"{COLLECTION}:d1")])
    index.search.side_effect = [batch, SimpleNamespace(docs=[])]

    vector.delete_by_metadata_field("annotation_id", "a-1")

    assert _query_string(index) == r"@annotation_id:{a\-1}"
    client.delete.assert_called_once_with(f"{COLLECTION}:d1")


def test_delete_by_metadata_field_rejects_unindexed_key(vector):
    with pytest.raises(ValueError, match="doc_hash"):
        vector.delete_by_metadata_field("doc_hash", "x")


def test_delete_drops_index_and_ignores_missing(client, vector):
    index = client.ft.return_value
    vector.delete()
    index.dropindex.assert_called_once_with(delete_documents=True)

    index.dropindex.side_effect = ResponseError("Unknown index name")
    vector.delete()

    index.dropindex.side_effect = ResponseError("boom")
    with pytest.raises(ResponseError):
        vector.delete()


def test_factory_sets_index_struct_for_new_dataset(client, monkeypatch):
    monkeypatch.setattr(redis_vector_module.dify_config, "REDIS_VECTOR_URL", "redis://localhost:6379")
    dataset = MagicMock(id="11111111-2222-3333-4444-555555555555", index_struct_dict=None)

    vector = RedisVectorFactory().init_vector(dataset, [], MagicMock())

    assert vector._collection_name == "Vector_index_11111111_2222_3333_4444_555555555555_Node"
    assert json.loads(dataset.index_struct) == {
        "type": "redis",
        "vector_store": {"class_prefix": vector._collection_name},
    }


def test_factory_reuses_existing_collection(client, monkeypatch):
    monkeypatch.setattr(redis_vector_module.dify_config, "REDIS_VECTOR_URL", "redis://localhost:6379")
    dataset = MagicMock(id="x", index_struct_dict={"vector_store": {"class_prefix": "Existing_Node"}})

    assert RedisVectorFactory().init_vector(dataset, [], MagicMock())._collection_name == "Existing_Node"

from dify_vdb_redis.redis_vector import RedisVector, RedisVectorConfig

from core.rag.datasource.vdb.vector_integration_test_support import (
    AbstractVectorTest,
)


class RedisVectorTest(AbstractVectorTest):
    def __init__(self):
        super().__init__()
        self.vector = RedisVector(
            collection_name=self.collection_name,
            config=RedisVectorConfig(url="redis://:difyai123456@localhost:6381"),
        )


def test_redis_vector():
    RedisVectorTest().run_all_tests()

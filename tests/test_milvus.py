import pytest
from pymilvus import Collection, connections, utility
from pymilvus import CollectionSchema, FieldSchema, DataType
from src.config import settings
import os

@pytest.fixture(scope="function")
def test_milvus_connection():
    """测试Milvus向量库连接和集合创建功能"""
    print(f"当前向量库配置: vector_store_type={settings.vector_store_type}, milvus_host={settings.milvus_host}, milvus_port={settings.milvus_port}")
    print(f"从.env加载的配置: VECTOR_STORE_TYPE={os.getenv('VECTOR_STORE_TYPE')}")
    print(f"settings对象类型: {type(settings)}")
    print(f"settings实际值: {settings.model_dump()}")
    if settings.vector_store_type != "milvus":
        pytest.skip("当前配置不是Milvus向量库，跳过测试")
    
    try:
        # 连接Milvus
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port
        )
        
        # 验证连接
        assert connections.has_connection("default")
        
        # 创建测试集合
        collection_name = getattr(settings, "milvus_collection_name", "test_collection")
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=128)
        ]
        schema = CollectionSchema(fields, "测试集合")
        collection = Collection(name=collection_name, schema=schema)
        
        # 验证集合创建
        assert utility.has_collection(collection_name)
        
        yield collection
        
        # 清理测试集合
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # 断开连接
        connections.disconnect("default")
    except Exception as e:
        pytest.fail(f"Milvus连接或集合创建失败: {str(e)}")


def test_milvus_connection_works(test_milvus_connection):
    """测试Milvus连接是否正常工作"""
    assert test_milvus_connection is not None
    assert isinstance(test_milvus_connection, Collection)
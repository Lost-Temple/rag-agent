import os
import pytest
import shutil
from src.config import settings
from pymilvus import Collection

# 测试文件路径
TEST_FILE_PATH = "test_document.txt"

@pytest.fixture(scope="session", autouse=True)
def setup_test_directories():
    """确保测试所需的目录存在"""
    # 确保测试目录存在
    os.makedirs(settings.original_documents_path, exist_ok=True)
    os.makedirs(settings.vector_store_path, exist_ok=True)
    
    yield
    
    # 测试会话结束后清理，但保留目录结构
    for filename in os.listdir(settings.original_documents_path):
        if filename != ".gitkeep":
            file_path = os.path.join(settings.original_documents_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

@pytest.fixture(scope="function")
def test_file():
    """创建测试文件并在测试后清理"""
    # 创建测试文件
    with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档，用于测试文件上传功能。")
    
    yield TEST_FILE_PATH
    
    # 清理测试文件
    if os.path.exists(TEST_FILE_PATH):
        os.remove(TEST_FILE_PATH)

@pytest.fixture(scope="function")
def cleanup_temp_files():
    """清理测试过程中可能产生的临时文件"""
    yield
    
    # 清理可能存在的临时文件
    for filename in os.listdir("."):
        if filename.startswith("temp_"):
            os.remove(filename)

@pytest.fixture(scope="function")
def test_milvus_connection():
    """测试Milvus向量库连接和集合创建功能"""
    if settings.vector_store_type != "milvus":
        pytest.skip("当前配置不是Milvus向量库，跳过测试")
    
    try:
        from pymilvus import connections, utility
        
        # 连接Milvus
        print(f"正在连接Milvus: host={settings.milvus_host}, port={settings.milvus_port}")
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port
        )
        
        # 验证连接
        print("验证Milvus连接状态...")
        assert connections.has_connection("default")
        print("Milvus连接验证成功")
        
        # 创建测试集合
        collection_name = getattr(settings, "milvus_collection_name", "test_collection")
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # 创建简单集合
        from pymilvus import CollectionSchema, FieldSchema, DataType
        
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=128)
        ]
        schema = CollectionSchema(fields, "测试集合")
        print(f"正在创建集合: {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        
        # 验证集合创建
        print(f"验证集合 {collection_name} 是否创建成功...")
        assert utility.has_collection(collection_name)
        print(f"集合 {collection_name} 创建验证成功")
        
        yield collection
        
        # 清理测试集合
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # 断开连接
        connections.disconnect("default")
    except Exception as e:
        pytest.fail(f"Milvus连接或集合创建失败: {str(e)}")
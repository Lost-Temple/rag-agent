import os
import pytest
import uuid
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_community.docstore.document import Document
from src.api.api_service import app
from src.config import settings
from src.models.vectorization.vectorizer import Vectorizer

# 创建测试客户端
client = TestClient(app)

# 测试文件路径
TEST_FILE_PATH = "test_document.txt"

@pytest.fixture(scope="function")
def setup_test_environment():
    """设置测试环境"""
    # 确保测试目录存在
    os.makedirs(settings.original_documents_path, exist_ok=True)
    os.makedirs(settings.vector_store_path, exist_ok=True)
    
    # 创建测试文件
    with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档，用于测试向量存储更新功能。")
    
    yield
    
    # 清理测试文件
    if os.path.exists(TEST_FILE_PATH):
        os.remove(TEST_FILE_PATH)

@pytest.mark.asyncio
async def test_vectorstore_update_after_upload(setup_test_environment):
    """测试文档上传后向量存储是否正确更新"""
    # 模拟UUID生成，使其返回固定值以便测试
    fixed_uuid = "12345678-1234-5678-1234-567812345678"
    
    # 创建模拟文档
    mock_documents = [
        Document(page_content="测试文档内容第一部分", metadata={"source": TEST_FILE_PATH}),
        Document(page_content="测试文档内容第二部分", metadata={"source": TEST_FILE_PATH})
    ]
    
    # 模拟向量存储
    mock_vector_store = MagicMock()
    
    with patch('uuid.uuid4', return_value=uuid.UUID(fixed_uuid)), \
         patch('src.api.api_service.rag_system.doc_processor.process_document', return_value=mock_documents), \
         patch('src.api.api_service.rag_system.vectorizer.vector_store', mock_vector_store), \
         patch('src.api.api_service.rag_system.vectorizer.initialize_vector_store') as mock_initialize_vector_store, \
         patch('src.api.api_service.rag_system.doc_processor.generate_document_summary', return_value="测试摘要"), \
         patch('src.api.api_service.rag_system.graph_store.create_document_node'), \
         patch('src.api.api_service.rag_system.graph_store.create_chunk_node'):
        
        # 发送上传请求
        with open(TEST_FILE_PATH, "rb") as f:
            response = client.post(
                "/documents/upload",
                files={"file": ("test_document.txt", f, "text/plain")}
            )
        
        # 验证响应状态码
        assert response.status_code == 200
        
        # 验证向量存储初始化函数是否被调用，以及传入的参数是否正确
        mock_initialize_vector_store.assert_called_once_with(mock_documents)

@pytest.mark.asyncio
async def test_vectorstore_initialization_with_empty_documents(setup_test_environment):
    """测试使用空文档列表初始化向量存储的情况"""
    # 创建Vectorizer实例
    vectorizer = Vectorizer()
    
    # 使用空文档列表初始化向量存储
    with patch('src.utils.logger.warning') as mock_logger_warning:
        vectorizer.initialize_vector_store([])
        
        # 验证是否记录了警告日志
        mock_logger_warning.assert_called_once_with("尝试使用空文档列表初始化向量库，不允许此操作")

@pytest.mark.asyncio
async def test_vectorstore_faiss_save_after_initialization(setup_test_environment):
    """测试FAISS向量存储初始化后是否正确保存"""
    # 模拟文档
    mock_documents = [
        Document(page_content="测试文档内容", metadata={"source": TEST_FILE_PATH})
    ]
    
    # 模拟FAISS向量存储
    mock_faiss_store = MagicMock()
    
    # 修改设置使用FAISS
    original_vector_store_type = settings.vector_store_type
    settings.vector_store_type = "faiss"
    
    try:
        with patch('langchain_community.vectorstores.faiss.FAISS.from_documents', return_value=mock_faiss_store) as mock_faiss_from_documents:
            # 创建Vectorizer实例并初始化向量存储
            vectorizer = Vectorizer()
            vectorizer.initialize_vector_store(mock_documents)
            
            # 验证FAISS.from_documents是否被调用
            mock_faiss_from_documents.assert_called_once()
            
            # 验证save_local是否被调用
            mock_faiss_store.save_local.assert_called_once_with(settings.vector_store_path)
    finally:
        # 恢复原始设置
        settings.vector_store_type = original_vector_store_type
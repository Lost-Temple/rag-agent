import os
import pytest
import uuid
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_community.docstore.document import Document
from src.api.api_service import app
from src.config import settings
from src.models.document_processor import DocumentProcessor
pytestmark = pytest.mark.asyncio

# 创建测试客户端
client = TestClient(app)

# 测试文件路径
TEST_FILE_PATH = "test_document.txt"

@pytest.fixture(scope="function")
def setup_test_environment():
    """设置测试环境"""
    # 确保测试目录存在
    os.makedirs(settings.original_documents_path, exist_ok=True)
    
    # 创建测试文件
    with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档，用于测试文档摘要生成功能。")
    
    yield
    
    # 清理测试文件
    if os.path.exists(TEST_FILE_PATH):
        os.remove(TEST_FILE_PATH)

@pytest.mark.asyncio
async def test_document_summary_generation(setup_test_environment):
    """测试文档摘要生成功能"""
    # 模拟UUID生成，使其返回固定值以便测试
    fixed_uuid = "12345678-1234-5678-1234-567812345678"
    
    # 创建模拟文档
    mock_documents = [
        Document(page_content="测试文档内容", metadata={"source": TEST_FILE_PATH})
    ]
    
    # 模拟摘要
    mock_summary = "这是测试文档的摘要"
    
    with patch('uuid.uuid4', return_value=uuid.UUID(fixed_uuid)), \
         patch('src.api.api_service.rag_system.doc_processor.load_document', return_value=mock_documents), \
         patch('src.api.api_service.rag_system.doc_processor.summarizer.summarize_documents', return_value=mock_summary), \
         patch('src.api.api_service.rag_system.doc_processor.sqlite_store.save_document_summary', return_value=True) as mock_save_summary, \
         patch('src.api.api_service.rag_system.doc_processor.process_document', return_value=mock_documents), \
         patch('src.api.api_service.rag_system.vectorizer.initialize_vector_store'), \
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
        response_data = response.json()
        assert response_data["summary_generated"] == True
        
        # 验证摘要是否被保存到数据库
        mock_save_summary.assert_called_once_with(fixed_uuid, "test_document.txt", mock_summary)

@pytest.mark.asyncio
async def test_document_summary_generation_failure(setup_test_environment):
    """测试文档摘要生成失败的情况"""
    # 模拟UUID生成
    fixed_uuid = "12345678-1234-5678-1234-567812345678"
    
    # 创建模拟文档
    mock_documents = [
        Document(page_content="测试文档内容", metadata={"source": TEST_FILE_PATH})
    ]
    
    with patch('uuid.uuid4', return_value=uuid.UUID(fixed_uuid)), \
         patch('src.api.api_service.rag_system.doc_processor.load_document', return_value=mock_documents), \
         patch('src.api.api_service.rag_system.doc_processor.summarizer.summarize_documents', return_value="摘要生成失败"), \
         patch('src.api.api_service.rag_system.doc_processor.process_document', return_value=mock_documents), \
         patch('src.api.api_service.rag_system.vectorizer.initialize_vector_store'), \
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
        response_data = response.json()
        assert response_data["summary_generated"] == False

@pytest.mark.asyncio
async def test_document_summary_direct_generation():
    """直接测试DocumentProcessor的摘要生成功能"""
    # 创建测试文件
    with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档，用于直接测试摘要生成功能。")
    
    try:
        # 创建模拟文档
        mock_documents = [
            Document(page_content="测试文档内容", metadata={"source": TEST_FILE_PATH})
        ]
        
        # 模拟摘要
        mock_summary = "这是测试文档的摘要"
        
        # 创建DocumentProcessor实例
        doc_processor = DocumentProcessor()
        
        with patch.object(doc_processor, 'load_document', return_value=mock_documents), \
             patch.object(doc_processor.summarizer, 'summarize_documents', return_value=mock_summary), \
             patch.object(doc_processor.sqlite_store, 'save_document_summary', return_value=True) as mock_save_summary:
            
            # 生成摘要
            doc_id = "test-doc-id"
            filename = "test_document.txt"
            summary = await doc_processor.generate_document_summary(TEST_FILE_PATH, doc_id, filename)
            
            # 验证结果
            assert summary == mock_summary
            mock_save_summary.assert_called_once_with(doc_id, filename, mock_summary)
    finally:
        # 清理测试文件
        if os.path.exists(TEST_FILE_PATH):
            os.remove(TEST_FILE_PATH)
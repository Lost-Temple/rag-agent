import pytest
import uuid
from unittest.mock import patch
from langchain_core.documents import Document
from src.models.document_processor import DocumentProcessor
from src.config import settings

pytestmark = pytest.mark.asyncio

@pytest.mark.asyncio
async def test_document_summary_generation(client, test_file, fixed_uuid, mock_documents):
    """测试文档摘要生成功能"""
    # 模拟摘要
    mock_summary = "这是测试文档的摘要"
    
    with patch('uuid.uuid4', return_value=uuid.UUID(fixed_uuid)), \
         patch('src.api.api_service.rag_system.doc_processor.load_document', return_value=mock_documents), \
         patch('src.api.api_service.rag_system.doc_processor.summarizer.summarize_documents', return_value=mock_summary), \
         patch('src.api.api_service.rag_system.doc_processor.store.save_document_summary', return_value=True) as mock_save_summary, \
         patch('src.api.api_service.rag_system.doc_processor.process_document', return_value=mock_documents), \
         patch('src.api.api_service.rag_system.vectorizer.initialize_vector_store'), \
         patch('src.api.api_service.rag_system.graph_store.create_document_node'), \
         patch('src.api.api_service.rag_system.graph_store.create_chunk_node'):
        
        # 发送上传请求
        with open(test_file, "rb") as f:
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
async def test_document_summary_generation_failure(client, test_file, fixed_uuid, mock_documents):
    """测试文档摘要生成失败的情况"""
    with patch('uuid.uuid4', return_value=uuid.UUID(fixed_uuid)), \
         patch('src.api.api_service.rag_system.doc_processor.load_document', return_value=mock_documents), \
         patch('src.api.api_service.rag_system.doc_processor.summarizer.summarize_documents', return_value="摘要生成失败"), \
         patch('src.api.api_service.rag_system.doc_processor.process_document', return_value=mock_documents), \
         patch('src.api.api_service.rag_system.vectorizer.initialize_vector_store'), \
         patch('src.api.api_service.rag_system.graph_store.create_document_node'), \
         patch('src.api.api_service.rag_system.graph_store.create_chunk_node'):
        
        # 发送上传请求
        with open(test_file, "rb") as f:
            response = client.post(
                "/documents/upload",
                files={"file": ("test_document.txt", f, "text/plain")}
            )
        
        # 验证响应状态码
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["summary_generated"] == False

@pytest.mark.asyncio
async def test_document_summary_direct_generation(test_file_with_content):
    """直接测试DocumentProcessor的摘要生成功能"""
    # 创建模拟文档
    mock_documents = [
        Document(page_content="测试文档内容", metadata={"source": test_file_with_content})
    ]
    
    # 模拟摘要
    mock_summary = "这是测试文档的摘要"
    
    # 创建DocumentProcessor实例
    doc_processor = DocumentProcessor()
    
    with patch.object(doc_processor, 'load_document', return_value=mock_documents), \
         patch.object(doc_processor.summarizer, 'summarize_documents', return_value=mock_summary), \
         patch.object(doc_processor.store, 'save_document_summary', return_value=True) as mock_save_summary:
        
        # 生成摘要
        doc_id = "test-doc-id"
        filename = "test_document.txt"
        summary = await doc_processor.generate_document_summary(test_file_with_content, doc_id, filename)
        
        # 验证结果
        assert summary == mock_summary
        mock_save_summary.assert_called_once_with(doc_id, filename, mock_summary)
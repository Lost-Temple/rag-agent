import pytest
from fastapi.testclient import TestClient
import uuid
import os
from unittest.mock import patch, MagicMock

from src.api.api_service import app, RAGSystem
from src.config import settings

client = TestClient(app)

@pytest.fixture
def mock_rag_system():
    """创建模拟的RAG系统组件"""
    with patch('src.api.api_service.rag_system') as mock_system:
        yield mock_system

def test_get_document_success(mock_rag_system):
    """测试成功获取文档的情况"""
    # 准备测试数据
    test_doc_id = str(uuid.uuid4())
    test_metadata = {
        "doc_id": test_doc_id,
        "filename": "test_document.pdf",
        "content_type": "application/pdf",
        "original_file_path": f"{settings.original_documents_path}/{test_doc_id}_test_document.pdf"
    }
    test_chunks = [
        {"chunk_id": f"{test_doc_id}_chunk_0", "content": "这是第一个文档片段", "metadata": {"chunk_index": 0}},
        {"chunk_id": f"{test_doc_id}_chunk_1", "content": "这是第二个文档片段", "metadata": {"chunk_index": 1}}
    ]
    test_summary = {"doc_id": test_doc_id, "summary": "这是测试文档的摘要"}
    
    # 设置模拟返回值
    mock_rag_system.graph_store.get_document_metadata.return_value = test_metadata
    mock_rag_system.graph_store.get_document_chunks.return_value = test_chunks
    mock_rag_system.doc_processor.get_document_summary.return_value = test_summary
    
    # 发送请求
    response = client.get(f"/documents/{test_doc_id}")
    
    # 验证结果
    assert response.status_code == 200
    assert response.json() == {
        "doc_id": test_doc_id,
        "metadata": test_metadata,
        "chunks": test_chunks,
        "summary": test_summary["summary"]
    }
    
    # 验证模拟函数被正确调用
    mock_rag_system.graph_store.get_document_metadata.assert_called_once_with(test_doc_id)
    mock_rag_system.graph_store.get_document_chunks.assert_called_once_with(test_doc_id)
    mock_rag_system.doc_processor.get_document_summary.assert_called_once_with(test_doc_id)

def test_get_document_not_found(mock_rag_system):
    """测试文档不存在的情况"""
    # 准备测试数据
    test_doc_id = str(uuid.uuid4())
    
    # 设置模拟返回值
    mock_rag_system.graph_store.get_document_metadata.return_value = None
    
    # 发送请求
    response = client.get(f"/documents/{test_doc_id}")
    
    # 验证结果 - 现在API返回404而不是500
    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}
    
    # 验证模拟函数被正确调用
    mock_rag_system.graph_store.get_document_metadata.assert_called_once_with(test_doc_id)
    mock_rag_system.graph_store.get_document_chunks.assert_not_called()
    mock_rag_system.doc_processor.get_document_summary.assert_not_called()

def test_get_document_no_summary(mock_rag_system):
    """测试文档存在但没有摘要的情况"""
    # 准备测试数据
    test_doc_id = str(uuid.uuid4())
    test_metadata = {
        "doc_id": test_doc_id,
        "filename": "test_document.pdf",
        "content_type": "application/pdf",
        "original_file_path": f"{settings.original_documents_path}/{test_doc_id}_test_document.pdf"
    }
    test_chunks = [
        {"chunk_id": f"{test_doc_id}_chunk_0", "content": "这是第一个文档片段", "metadata": {"chunk_index": 0}}
    ]
    
    # 设置模拟返回值
    mock_rag_system.graph_store.get_document_metadata.return_value = test_metadata
    mock_rag_system.graph_store.get_document_chunks.return_value = test_chunks
    mock_rag_system.doc_processor.get_document_summary.return_value = None
    
    # 发送请求
    response = client.get(f"/documents/{test_doc_id}")
    
    # 验证结果
    assert response.status_code == 200
    assert response.json() == {
        "doc_id": test_doc_id,
        "metadata": test_metadata,
        "chunks": test_chunks,
        "summary": None
    }

def test_get_document_server_error(mock_rag_system):
    """测试服务器错误的情况"""
    # 准备测试数据
    test_doc_id = str(uuid.uuid4())
    
    # 设置模拟返回值抛出异常
    mock_rag_system.graph_store.get_document_metadata.side_effect = Exception("Database connection error")
    
    # 发送请求
    response = client.get(f"/documents/{test_doc_id}")
    
    # 验证结果
    assert response.status_code == 500
    assert response.json() == {"detail": "Database connection error"}
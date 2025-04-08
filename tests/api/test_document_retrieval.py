import pytest
from unittest.mock import patch, MagicMock

# 移除这一行，因为这些测试函数不是异步的
# pytestmark = pytest.mark.asyncio

def test_get_document_success(client, fixed_uuid):
    """测试成功获取文档的情况"""
    # 准备测试数据
    test_metadata = {
        "doc_id": fixed_uuid,
        "filename": "test_document.pdf",
        "content_type": "application/pdf",
        "original_file_path": f"path/to/{fixed_uuid}_test_document.pdf"
    }
    test_chunks = [
        {"chunk_id": f"{fixed_uuid}_chunk_0", "content": "这是第一个文档片段", "metadata": {"chunk_index": 0}},
        {"chunk_id": f"{fixed_uuid}_chunk_1", "content": "这是第二个文档片段", "metadata": {"chunk_index": 1}}
    ]
    test_summary = {"doc_id": fixed_uuid, "summary": "这是测试文档的摘要"}
    
    # 设置模拟返回值
    with patch('src.api.api_service.rag_system.graph_store.get_document_metadata', return_value=test_metadata), \
         patch('src.api.api_service.rag_system.graph_store.get_document_chunks', return_value=test_chunks), \
         patch('src.api.api_service.rag_system.doc_processor.get_document_summary', return_value=test_summary):
        
        # 发送请求
        response = client.get(f"/documents/{fixed_uuid}")
        
        # 验证结果
        assert response.status_code == 200
        assert response.json() == {
            "doc_id": fixed_uuid,
            "metadata": test_metadata,
            "chunks": test_chunks,
            "summary": test_summary["summary"]
        }

def test_get_document_not_found(client, fixed_uuid):
    """测试文档不存在的情况"""
    # 设置模拟返回值
    with patch('src.api.api_service.rag_system.graph_store.get_document_metadata', return_value=None):
        
        # 发送请求
        response = client.get(f"/documents/{fixed_uuid}")
        
        # 验证结果 - 现在API返回404而不是500
        assert response.status_code == 404
        assert response.json() == {"detail": "Document not found"}

def test_get_document_no_summary(client, fixed_uuid):
    """测试文档存在但没有摘要的情况"""
    # 准备测试数据
    test_metadata = {
        "doc_id": fixed_uuid,
        "filename": "test_document.pdf",
        "content_type": "application/pdf",
        "original_file_path": f"path/to/{fixed_uuid}_test_document.pdf"
    }
    test_chunks = [
        {"chunk_id": f"{fixed_uuid}_chunk_0", "content": "这是第一个文档片段", "metadata": {"chunk_index": 0}}
    ]
    
    # 设置模拟返回值
    with patch('src.api.api_service.rag_system.graph_store.get_document_metadata', return_value=test_metadata), \
         patch('src.api.api_service.rag_system.graph_store.get_document_chunks', return_value=test_chunks), \
         patch('src.api.api_service.rag_system.doc_processor.get_document_summary', return_value=None):
        
        # 发送请求
        response = client.get(f"/documents/{fixed_uuid}")
        
        # 验证结果
        assert response.status_code == 200
        assert response.json() == {
            "doc_id": fixed_uuid,
            "metadata": test_metadata,
            "chunks": test_chunks,
            "summary": None
        }

def test_get_all_document_summaries(client):
    """测试获取所有文档摘要"""
    # 准备测试数据
    test_summaries = [
        {"doc_id": "id1", "filename": "doc1.pdf", "summary": "摘要1"},
        {"doc_id": "id2", "filename": "doc2.pdf", "summary": "摘要2"}
    ]
    
    # 设置模拟返回值
    with patch('src.api.api_service.rag_system.doc_processor.get_all_document_summaries', return_value=test_summaries):
        
        # 发送请求
        response = client.get("/documents/summaries/all")
        
        # 验证结果
        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "count": len(test_summaries),
            "summaries": test_summaries
        }
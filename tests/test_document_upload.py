import os
import pytest
import uuid
import shutil
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
from src.api.api_service import app, RAGSystem
from src.config import settings

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
        f.write("这是一个测试文档，用于测试文件上传功能。")
    
    yield
    
    # 清理测试文件
    if os.path.exists(TEST_FILE_PATH):
        os.remove(TEST_FILE_PATH)
    
    # 清理测试过程中创建的文件
    for filename in os.listdir(settings.original_documents_path):
        if filename != ".gitkeep":
            file_path = os.path.join(settings.original_documents_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

@pytest.mark.asyncio
async def test_upload_document(setup_test_environment):
    """测试文档上传功能"""
    # 模拟UUID生成，使其返回固定值以便测试
    fixed_uuid = "12345678-1234-5678-1234-567812345678"
    with patch('uuid.uuid4', return_value=uuid.UUID(fixed_uuid)):
        # 模拟文档处理器和向量化器
        with patch('src.api.api_service.rag_system.doc_processor.process_document') as mock_process_document, \
             patch('src.api.api_service.rag_system.vectorizer.initialize_vector_store') as mock_initialize_vector_store, \
             patch('src.api.api_service.rag_system.doc_processor.generate_document_summary') as mock_generate_summary, \
             patch('src.api.api_service.rag_system.graph_store.create_document_node') as mock_create_document_node, \
             patch('src.api.api_service.rag_system.graph_store.create_chunk_node') as mock_create_chunk_node:
            
            # 模拟处理后的文档
            from langchain_community.docstore.document import Document
            mock_documents = [
                Document(page_content="测试文档内容第一部分", metadata={"source": TEST_FILE_PATH}),
                Document(page_content="测试文档内容第二部分", metadata={"source": TEST_FILE_PATH})
            ]
            mock_process_document.return_value = mock_documents
            mock_generate_summary.return_value = "这是测试文档的摘要"
            
            # 打开测试文件并发送上传请求
            with open(TEST_FILE_PATH, "rb") as f:
                response = client.post(
                    "/documents/upload",
                    files={"file": ("test_document.txt", f, "text/plain")}
                )
            
            # 验证响应状态码和内容
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "success"
            assert response_data["doc_id"] == fixed_uuid
            assert response_data["chunks_count"] == len(mock_documents)
            assert response_data["summary_generated"] == True
            
            # 验证文件是否保存到原始文档目录
            expected_file_path = os.path.join(
                settings.original_documents_path, 
                f"{fixed_uuid}_test_document.txt"
            )
            assert os.path.exists(expected_file_path)
            
            # 验证各个模拟函数是否被正确调用
            mock_process_document.assert_called_once()
            mock_initialize_vector_store.assert_called_once_with(mock_documents)
            mock_generate_summary.assert_called_once()
            mock_create_document_node.assert_called_once()
            assert mock_create_chunk_node.call_count == len(mock_documents)

@pytest.mark.asyncio
async def test_upload_document_error_handling(setup_test_environment):
    """测试文档上传错误处理"""
    # 模拟文档处理器抛出异常
    with patch('src.api.api_service.rag_system.doc_processor.process_document', side_effect=Exception("处理文档时出错")):
        # 打开测试文件并发送上传请求
        with open(TEST_FILE_PATH, "rb") as f:
            response = client.post(
                "/documents/upload",
                files={"file": ("test_document.txt", f, "text/plain")}
            )
        
        # 验证响应状态码和错误信息
        assert response.status_code == 500
        response_data = response.json()
        assert "detail" in response_data
        assert "处理文档时出错" in response_data["detail"]

@pytest.mark.asyncio
async def test_upload_unsupported_file_type(setup_test_environment):
    """测试上传不支持的文件类型"""
    # 创建一个不支持的文件类型
    unsupported_file = "test_unsupported.xyz"
    with open(unsupported_file, "w") as f:
        f.write("这是一个不支持的文件类型")
    
    try:
        # 模拟文档处理器抛出不支持的文件类型异常
        with patch('src.api.api_service.rag_system.doc_processor.process_document', 
                  side_effect=ValueError("Unsupported file type: xyz")):
            # 发送上传请求
            with open(unsupported_file, "rb") as f:
                response = client.post(
                    "/documents/upload",
                    files={"file": ("test_unsupported.xyz", f, "application/octet-stream")}
                )
            
            # 验证响应状态码和错误信息
            assert response.status_code == 500
            response_data = response.json()
            assert "detail" in response_data
            assert "Unsupported file type" in response_data["detail"]
    finally:
        # 清理测试文件
        if os.path.exists(unsupported_file):
            os.remove(unsupported_file)
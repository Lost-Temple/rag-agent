import os
import pytest
import shutil
from src.config import settings

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
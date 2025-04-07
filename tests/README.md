# RAG Agent 测试用例

本目录包含了RAG Agent系统的测试用例，主要测试文件上传流程及相关功能。

## 测试内容

1. **文档上传测试** (`test_document_upload.py`)
   - 测试文件上传到`/documents/upload`端点的功能
   - 验证响应状态码和内容
   - 检查文件是否正确存储在`original_documents`目录
   - 测试错误处理和不支持的文件类型

2. **向量存储更新测试** (`test_vectorstore_update.py`)
   - 测试文件上传后向量存储是否正确更新
   - 测试空文档列表初始化向量存储的情况
   - 测试FAISS向量存储初始化和保存

3. **文档摘要生成测试** (`test_document_summary.py`)
   - 测试文件上传后是否正确生成摘要
   - 测试摘要生成失败的情况
   - 直接测试DocumentProcessor的摘要生成功能

## 运行测试

### 安装测试依赖

```bash
pip install pytest pytest-asyncio pytest-mock
```

### 运行所有测试

```bash
python -m pytest tests/
```

### 运行特定测试文件

```bash
python -m pytest tests/test_document_upload.py
```

### 运行特定测试函数

```bash
python -m pytest tests/test_document_upload.py::test_upload_document
```

## 测试配置

测试配置在`conftest.py`文件中定义，包括：

- 测试目录的设置和清理
- 测试文件的创建和清理
- 临时文件的清理

## 注意事项

1. 测试会创建临时文件和目录，测试结束后会自动清理
2. 测试使用模拟对象(mock)来隔离依赖，不会影响实际的数据库和向量存储
3. 测试会使用固定的UUID，以便于验证结果
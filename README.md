# RAG 系统

一个基于Langchain的可配置RAG（检索增强生成）系统，支持文档处理、向量化、混合检索等功能，使用Neo4j作为图数据库存储，并提供RESTful API和MCP远程调用接口。支持使用Ollama本地大模型进行文档处理和问答。

## 功能特点

- 支持多种文档格式（PDF、DOCX、TXT、MD）的处理和切分
- 使用先进的向量化模型进行文档嵌入
- 支持混合检索策略（向量检索 + 重排序）
- 使用Neo4j图数据库存储文档关系和元数据
- 提供RESTful API接口用于文档上传和管理
- 支持通过MCP进行远程检索调用
- 支持使用Ollama本地大模型进行文档嵌入和问答
- 高度可配置，支持自定义模型和参数

## 安装

1. 克隆项目并安装依赖：

```bash
pip install -r requirements.txt
```

2. 安装并启动Neo4j数据库

3. 创建配置文件：

将示例配置文件复制为 `.env`，并根据需要修改配置：

```bash
cp .env.example .env
```

## 配置

主要配置项（在 `.env` 文件中设置）：

- `EMBEDDING_MODEL`：向量化模型名称
- `RERANKER_MODEL`：重排序模型名称
- `NEO4J_URI`：Neo4j数据库连接URI
- `NEO4J_USER`：Neo4j用户名
- `NEO4J_PASSWORD`：Neo4j密码
- `VECTOR_STORE_TYPE`：向量存储类型（chroma/faiss/milvus）
- `VECTOR_STORE_PATH`：向量存储路径
- `ORIGINAL_DOCUMENTS_PATH`：原始文档存储路径
- `MILVUS_HOST`：Milvus服务器地址
- `MILVUS_PORT`：Milvus服务器端口
- `MILVUS_USER`：Milvus用户名
- `MILVUS_PASSWORD`：Milvus密码
- `MILVUS_COLLECTION`：Milvus集合名称
- `MILVUS_INDEX_TYPE`：Milvus索引类型（IVF_FLAT/HNSW/FLAT等）
- `MILVUS_METRIC_TYPE`：Milvus度量类型（IP/L2等）
- `MILVUS_INDEX_PARAMS`：Milvus索引参数（JSON格式，如{"nlist": 1024}）
- `MILVUS_SEARCH_PARAMS`：Milvus搜索参数（JSON格式，如{"nprobe": 16}）
- `API_PORT`：RESTful API服务端口
- `MCP_PORT`：MCP服务端口
- `USE_OLLAMA`：是否使用Ollama模型（true/false）
- `OLLAMA_BASE_URL`：Ollama服务地址
- `OLLAMA_MODEL`：Ollama LLM模型名称
- `OLLAMA_EMBEDDING_MODEL`：Ollama嵌入模型名称

## 使用方法

### 1. 启动服务

通过main.py统一入口启动所有服务：

```bash
python main.py
```

支持以下可选参数：
- --port: 指定服务端口（默认8000）
- --debug: 启用调试模式

### 2. 上传文档

使用RESTful API上传文档：

```bash
curl -X POST "http://localhost:8000/documents/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```

### 3. 检索文档

使用MCP客户端进行检索：

```python
from mcp_service import RetrievalClient

client = RetrievalClient()
results = client.search("your query here", top_k=5)
for result in results:
    print(f"Content: {result['content']}")
    print(f"Score: {result['final_score']}")
```

### 4. 使用Ollama进行问答

通过API进行问答：

```bash
curl -X POST "http://localhost:8000/llm/question" \
     -H "Content-Type: application/json" \
     -d '{"question":"你的问题", "top_k":5}'
```

或者使用Python客户端：

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/llm/question",
    json={"question": "你的问题", "top_k": 5}
)

result = response.json()
print(f"回答: {result['answer']}")
```

## API文档

RESTful API文档可在服务启动后访问：

```
http://localhost:8000/docs
```

## 许可证


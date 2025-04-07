from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # 向量化模型配置
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Ollama配置
    use_ollama: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # Neo4j配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # 向量数据库配置
    vector_store_type: str = "chroma"  # 支持chroma, faiss, milvus等
    vector_store_path: str = "./data/vectorstore"
    
    # 原始文档存储配置
    original_documents_path: str = "./data/original_documents"
    
    # SQLite数据库配置
    sqlite_db_path: str = "./data/database/document_summaries.db"
    
    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_collection: str = "document_store"
    
    # Milvus索引配置
    milvus_index_type: str = "IVF_FLAT"  # 支持IVF_FLAT, HNSW, FLAT等
    milvus_metric_type: str = "IP"  # 支持IP(内积), L2(欧氏距离)等
    milvus_index_params: dict = {
        "nlist": 1024  # IVF_FLAT的聚类中心数量
    }
    milvus_search_params: dict = {
        "nprobe": 16  # 搜索时探测的聚类中心数量
    }
    
    # 文档处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # API服务配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # MCP服务配置
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 50051
    
    # 日志配置
    log_file_max_bytes: int = 10*1024*1024  # 日志文件大小限制，默认10MB
    log_file_backup_count: int = 5         # 保留的备份文件数量
    
    model_config = {
        "env_file": ".env"
    }

settings = Settings()
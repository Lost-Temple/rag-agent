from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Milvus
from pymilvus import connections, Collection, utility
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from src.config import settings
from src.models.llm.ollama_llm import OllamaLLMClient
from src.utils import logger
import os

class Vectorizer:
    def __init__(self):
        # 检查是否使用Ollama嵌入模型
        if settings.use_ollama:
            ollama_client = OllamaLLMClient()
            self.embedding_model = ollama_client.get_embedding_model()
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=settings.embedding_model
            )
        
        self.reranker = CrossEncoder(settings.reranker_model)
        self.vector_store = None
        
    def initialize_vector_store(self, documents: List[Document]):
        """初始化向量数据库"""
        # 确保有文档才初始化，避免空向量库的维度问题
        if not documents:
            logger.warning("尝试使用空文档列表初始化向量库，不允许此操作")
            return
            
        if settings.vector_store_type.lower() == "chroma":
            # 确保目录存在
            if not os.path.exists(settings.vector_store_path):
                os.makedirs(settings.vector_store_path)
                
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=settings.vector_store_path
            )
        elif settings.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )
            if not os.path.exists(settings.vector_store_path):
                os.makedirs(settings.vector_store_path)
            self.vector_store.save_local(settings.vector_store_path)
        elif settings.vector_store_type.lower() == "milvus":
            logger.info(f"正在初始化Milvus向量库，连接到 {settings.milvus_host}:{settings.milvus_port}，集合名称: {settings.milvus_collection}")
            try:
                self.vector_store = Milvus.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    collection_name=settings.milvus_collection,
                    connection_args={
                        "host": settings.milvus_host,
                        "port": settings.milvus_port,
                        "user": settings.milvus_user,
                        "password": settings.milvus_password
                    },
                    index_params={
                        "index_type": settings.milvus_index_type,
                        "metric_type": settings.milvus_metric_type,
                        "params": settings.milvus_index_params
                    },
                    search_params=settings.milvus_search_params
                )
                logger.info(f"成功初始化Milvus向量库，已存储{len(documents)}个文档")
            except Exception as e:
                logger.error(f"Milvus向量库初始化失败: {str(e)}，将回退到使用FAISS")
                # 回退到FAISS
                logger.info("正在回退到FAISS向量库...")
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embedding_model
                )
                if not os.path.exists(settings.vector_store_path):
                    os.makedirs(settings.vector_store_path)
                self.vector_store.save_local(settings.vector_store_path)
                logger.info(f"已成功回退到FAISS向量库并保存到{settings.vector_store_path}")
        logger.info(f"成功初始化向量库，类型: {settings.vector_store_type}，文档数量: {len(documents)}")
        
    def check_milvus_connection(self) -> bool:
        """仅检查Milvus连接是否正常，不创建collection
        
        Returns:
            bool: 连接是否成功
        """
        logger.info(f"正在检查Milvus连接: {settings.milvus_host}:{settings.milvus_port}")
        try:
            # 尝试连接Milvus服务器
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
                user=settings.milvus_user,
                password=settings.milvus_password
            )
            
            # 验证连接是否成功
            connected = connections.has_connection("default")
            if connected:
                logger.info("Milvus连接成功")
                # 断开连接
                connections.disconnect("default")
                return True
            else:
                logger.error("Milvus连接失败")
                return False
        except Exception as e:
            logger.error(f"Milvus连接检查失败: {str(e)}")
            return False
    
    def ensure_vector_store(self, documents: List[Document]) -> bool:
        """确保向量库已初始化，如果未初始化则使用提供的文档初始化
        
        Args:
            documents: 用于初始化向量库的文档列表
            
        Returns:
            bool: 向量库是否可用
        """
        if self.vector_store is None:
            if documents:
                logger.info(f"向量库未初始化，正在使用提供的文档进行初始化...文档数量: {len(documents)}")
                logger.info(f"当前配置的向量库类型: {settings.vector_store_type}")
                
                # 对于Milvus，只有在有文档需要存储时才创建collection
                if settings.vector_store_type.lower() == "milvus":
                    logger.info(f"Milvus配置 - 主机: {settings.milvus_host}, 端口: {settings.milvus_port}, 集合: {settings.milvus_collection}")
                    # 先检查连接是否正常
                    if not self.check_milvus_connection():
                        logger.error("Milvus连接失败，无法初始化向量库")
                        return False
                    logger.info("Milvus连接正常，正在创建集合并存储文档...")
                
                # 初始化向量库
                self.initialize_vector_store(documents)
                
                # 验证向量库是否成功初始化
                if self.vector_store is None:
                    logger.error("向量库初始化失败，仍为None")
                    return False
                else:
                    logger.info(f"向量库初始化成功，类型: {type(self.vector_store).__name__}")
                    return True
            else:
                logger.error("向量库未初始化且没有提供文档进行初始化")
                return False
        return True

    def is_initialized(self):
        """
        检查向量库是否已初始化

        Returns:
            bool: 如果向量库已初始化则返回True，否则返回False
        """
        # 根据实际的向量库实现来检查初始化状态
        # 例如，检查self.vector_store是否为None
        return hasattr(self, 'vector_store') and self.vector_store is not None
    
    def load_vector_store(self):
        """加载现有的向量数据库"""
        try:
            if settings.vector_store_type.lower() == "chroma":
                # 检查Chroma向量库目录是否存在
                if os.path.exists(settings.vector_store_path) and os.path.isdir(settings.vector_store_path) and len(os.listdir(settings.vector_store_path)) > 0:
                    self.vector_store = Chroma(
                        persist_directory=settings.vector_store_path,
                        embedding_function=self.embedding_model
                    )
                else:
                    # 不立即初始化，只在需要时创建
                    logger.info(f"Chroma向量库目录不存在或为空: {settings.vector_store_path}，将在需要时创建")
                    self.vector_store = None
            elif settings.vector_store_type.lower() == "faiss":
                # 检查FAISS索引文件是否存在
                faiss_index_path = os.path.join(settings.vector_store_path, "index.faiss")
                faiss_docstore_path = os.path.join(settings.vector_store_path, "index.pkl")
                
                if os.path.exists(faiss_index_path) and os.path.exists(faiss_docstore_path):
                    self.vector_store = FAISS.load_local(
                        settings.vector_store_path,
                        self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                else:
                    # 不立即初始化，只在需要时创建
                    logger.info(f"FAISS索引文件不存在: {settings.vector_store_path}，将在需要时创建")
                    self.vector_store = None
            elif settings.vector_store_type.lower() == "milvus":
                logger.info(f"正在尝试连接Milvus向量库: {settings.milvus_host}:{settings.milvus_port}")
                try:
                    # 尝试连接Milvus服务器
                    connections.connect(
                        alias="default",
                        host=settings.milvus_host,
                        port=settings.milvus_port,
                        user=settings.milvus_user,
                        password=settings.milvus_password
                    )
                    
                    # 验证连接是否成功
                    connected = connections.has_connection("default")
                    if connected:
                        logger.info("Milvus连接成功")
                        
                        # 检查集合是否存在
                        if utility.has_collection(settings.milvus_collection):
                            logger.info(f"Milvus集合 {settings.milvus_collection} 已存在，正在加载...")
                            
                            # 初始化Milvus向量库对象
                            self.vector_store = Milvus(
                                embedding_function=self.embedding_model,
                                collection_name=settings.milvus_collection,
                                connection_args={
                                    "host": settings.milvus_host,
                                    "port": settings.milvus_port,
                                    "user": settings.milvus_user,
                                    "password": settings.milvus_password
                                },
                                search_params=settings.milvus_search_params
                            )
                            logger.info(f"成功加载Milvus集合 {settings.milvus_collection}")
                        else:
                            logger.info(f"Milvus集合 {settings.milvus_collection} 不存在，将在需要时创建")
                            self.vector_store = None
                            if connections.has_connection("default"):
                                connections.disconnect("default")
                    else:
                        logger.error("Milvus连接失败")
                        self.vector_store = None
                except Exception as e:
                    logger.error(f"Milvus连接或加载失败: {str(e)}")
                    self.vector_store = None
        except Exception as e:
            logger.error(f"加载向量库失败: {str(e)}，将在需要时创建")
            # 确保向量存储目录存在
            if not os.path.exists(settings.vector_store_path):
                os.makedirs(settings.vector_store_path)
            # 不立即初始化空的向量库
            self.vector_store = None
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """混合检索：结合向量检索和重排序"""
        # 检查向量库是否已初始化
        if self.vector_store is None:
            logger.warning("向量库尚未初始化，无法执行检索操作")
            return []
            
        # 向量检索
        results = self.vector_store.similarity_search_with_score(query, k=k*2)
        
        # 准备重排序
        candidates = []
        for doc, score in results:
            candidates.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "vector_score": score
            })
        
        # 重排序
        pairs = [[query, doc["content"]] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        # 合并分数并排序
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(rerank_scores[i])
            # 综合分数计算（可以根据需要调整权重）
            doc["final_score"] = 0.3 * (1 - doc["vector_score"]) + 0.7 * doc["rerank_score"]
        
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:k]
        
    def get_document_chunks(self, document_path: str) -> List[Dict[str, Any]]:
        """获取同一原始文档的所有chunk及其元数据"""
        # 检查向量库是否已初始化
        if self.vector_store is None:
            logger.warning("向量库尚未初始化，无法获取文档块")
            return []
        
        # 准备结果列表
        chunks = []
        
        # 根据不同向量库类型使用最优的过滤查询方式
        if settings.vector_store_type.lower() == "milvus":
            # 使用Milvus的过滤查询功能
            filter_expr = f"metadata['source'] == '{document_path}'"
            results = self.vector_store.get(
                filter=filter_expr,
                output_fields=["*"],
                limit=10000  # 设置足够大的限制
            )
            
            for doc in results:
                chunks.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        elif settings.vector_store_type.lower() == "chroma":
            # 使用Chroma的where过滤功能
            try:
                # 尝试使用where参数进行过滤
                results = self.vector_store.get(
                    where={"source": document_path},
                    limit=10000
                )
                
                for doc in results:
                    chunks.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
            except Exception as e:
                logger.warning(f"使用Chroma的where过滤失败: {str(e)}，回退到全量获取后过滤")
                # 回退到获取所有文档后过滤
                all_docs = self.vector_store.get()
                for doc in all_docs:
                    if doc.metadata.get("source") == document_path:
                        chunks.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
        elif settings.vector_store_type.lower() == "faiss":
            # FAISS没有直接的过滤API，使用similarity_search的filter参数
            try:
                # 尝试使用空查询和metadata_filter进行过滤
                # 注意：这里使用一个空字符串作为查询，主要是为了使用filter功能
                results = self.vector_store.similarity_search(
                    "",  # 空查询
                    k=10000,  # 设置足够大的限制
                    filter={"source": document_path}
                )
                
                for doc in results:
                    chunks.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
            except Exception as e:
                logger.warning(f"使用FAISS的filter过滤失败: {str(e)}，回退到全量获取后过滤")
                # 回退到获取所有文档后过滤
                all_docs = self.vector_store.get()
                for doc in all_docs:
                    if doc.metadata.get("source") == document_path:
                        chunks.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
        else:
            # 对于其他向量库的简单实现，获取所有文档后过滤
            all_docs = self.vector_store.get()
            for doc in all_docs:
                if doc.metadata.get("source") == document_path:
                    chunks.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
        
        return chunks
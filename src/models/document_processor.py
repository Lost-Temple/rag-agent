from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_unstructured import UnstructuredLoader
from langchain_community.docstore.document import Document
from src.config import settings
from src.models.storage.peewee_store import PeeweeStore
from src.models.summarization.document_summarizer import DocumentSummarizer

class DocumentProcessor:
    def __init__(self):
        # 使用针对中文优化的分块设置
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        self.loader_map = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".doc": UnstructuredLoader,
            ".md": UnstructuredMarkdownLoader
        }
        
        # 初始化文档摘要生成器
        self.summarizer = DocumentSummarizer()
        
        # 初始化SQLite存储
        self.sqlite_store = PeeweeStore()
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载文档并返回Document对象列表"""
        file_extension = file_path.split(".")[-1].lower()
        loader_class = self.loader_map.get(f".{file_extension}")
        
        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
        loader = loader_class(file_path)
        documents = loader.load()
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档切分成较小的片段"""
        return self.text_splitter.split_documents(documents)
    
    def process_document(self, file_path: str) -> List[Document]:
        """处理文档：加载并切分"""
        documents = self.load_document(file_path)
        return self.split_documents(documents)
    
    def extract_metadata(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """提取文档元数据"""
        return [doc.metadata for doc in documents]
    
    async def generate_document_summary(self, file_path: str, doc_id: str, filename: str) -> Optional[str]:
        """
        生成文档摘要并存储到SQLite数据库
        
        Args:
            file_path: 文档路径
            doc_id: 文档ID
            filename: 文件名
            
        Returns:
            Optional[str]: 生成的摘要，如果失败则返回None
        """
        try:
            # 加载原始文档（不分块）
            documents = self.load_document(file_path)
            
            # 生成摘要
            summary = await self.summarizer.summarize_documents(documents)
            
            # 存储摘要到SQLite
            if summary and summary != "摘要生成失败" and summary != "多文档摘要生成失败":
                success = self.sqlite_store.save_document_summary(doc_id, filename, summary)
                if success:
                    return summary
            return None
        except Exception as e:
            from src.utils import logger
            logger.error(f"生成文档摘要失败: {str(e)}")
            return None
    
    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档摘要
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Optional[Dict[str, Any]]: 文档摘要信息，如果不存在则返回None
        """
        return self.sqlite_store.get_document_summary(doc_id)
    
    def get_all_document_summaries(self) -> List[Dict[str, Any]]:
        """
        获取所有文档摘要
        
        Returns:
            List[Dict[str, Any]]: 所有文档摘要的列表
        """
        return self.sqlite_store.get_all_document_summaries()
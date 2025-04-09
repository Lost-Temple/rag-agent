from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Type
from peewee import Database, Model, CharField, TextField, DateTimeField
from datetime import datetime
from src.utils import logger

class BaseDocumentStore(ABC):
    def __init__(self, db: Database):
        self.db = db
        self.db.connect(reuse_if_open=True)
        
        # 创建模型类
        self.BaseModel = self._create_base_model()
        self.DocumentSummary = self._create_document_summary_model()
        
        try:
            self.db.create_tables([self.DocumentSummary])
        except Exception as e:
            raise
    
    def _create_base_model(self) -> Type[Model]:
        """创建基础模型类"""
        class BaseModel(Model):
            class Meta:
                database = self.db
        return BaseModel
    
    def _create_document_summary_model(self) -> Type[Model]:
        """创建文档摘要模型类"""
        class DocumentSummary(self.BaseModel):
            doc_id = CharField(primary_key=True)
            filename = CharField()
            summary = TextField()
            created_at = DateTimeField(default=datetime.now)
            updated_at = DateTimeField(default=datetime.now)
            
            class Meta:
                table_name = 'document_summaries'
        return DocumentSummary
    
    def close(self):
        if not self.db.is_closed():
            self.db.close()
    
    @property
    def connection_context(self):
        return self.db.connection_context()

    @abstractmethod
    def save_document_summary(self, doc_id: str, filename: str, summary: str) -> bool:
        pass
    
    @abstractmethod
    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_all_document_summaries(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_paginated_summaries(self, page: int, page_size: int) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_summaries_count(self) -> int:
        pass
from peewee import MySQLDatabase, Model, CharField, TextField, DateTimeField
from typing import Optional, Dict, Any, List
from datetime import datetime
from src.config import settings
from src.utils import logger
from .base_store import BaseDocumentStore

class MySQLStore(BaseDocumentStore):
    def __init__(self):
        # 创建数据库连接
        db = MySQLDatabase(
            settings.mysql_db_name,
            user=settings.mysql_user,
            password=settings.mysql_password,
            host=settings.mysql_host,
            port=settings.mysql_port
        )
        # 调用父类初始化
        super().__init__(db)
        
        # 定义模型类
        class BaseModel(Model):
            class Meta:
                database = self.db
        
        # 定义文档摘要模型
        class DocumentSummary(BaseModel):
            doc_id = CharField(primary_key=True)
            filename = CharField()
            summary = TextField()
            created_at = DateTimeField(default=datetime.now)
            updated_at = DateTimeField(default=datetime.now)
            
            class Meta:
                table_name = 'document_summaries'
        
        # 保存模型类的引用
        self.DocumentSummary = DocumentSummary
        
        try:
            # 创建表
            self.db.create_tables([DocumentSummary])
            logger.info(f"成功初始化MySQL数据库: {settings.mysql_db_name}")
        except Exception as e:
            logger.error(f"初始化MySQL数据库失败: {str(e)}")
            raise
    
    # 使用装饰器的简化方式
    def save_document_summary(self, doc_id: str, filename: str, summary: str) -> bool:
        with self.db.connection_context():
            try:
                self.DocumentSummary.insert(
                    doc_id=doc_id,
                    filename=filename,
                    summary=summary
                ).on_conflict(
                    conflict_target=[self.DocumentSummary.doc_id],
                    update={
                        self.DocumentSummary.summary: summary,
                        self.DocumentSummary.updated_at: datetime.now()
                    }
                ).execute()
                return True
            except Exception as e:
                logger.error(f"保存文档摘要失败: {str(e)}")
                return False

    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self.db.connection_context():
            try:
                summary = self.DocumentSummary.get_or_none(self.DocumentSummary.doc_id == doc_id)
                if summary:
                    return {
                        'doc_id': summary.doc_id,
                        'filename': summary.filename,
                        'summary': summary.summary,
                        'created_at': summary.created_at,
                        'updated_at': summary.updated_at
                    }
                return None
            except Exception as e:
                logger.error(f"获取文档摘要失败: {str(e)}")
                return None


    def get_all_document_summaries(self) -> List[Dict[str, Any]]:
        """获取所有文档摘要(不推荐使用，建议使用分页查询)"""
        with self.db.connection_context():
            try:
                return [{
                    'doc_id': s.doc_id,
                    'filename': s.filename,
                    'summary': s.summary,
                    'created_at': s.created_at,
                    'updated_at': s.updated_at
                } for s in self.DocumentSummary.select()]
            except Exception as e:
                logger.error(f"获取所有文档摘要失败: {str(e)}")
                return []

    def get_paginated_summaries(self, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """分页获取文档摘要"""
        with self.db.connection_context():
            try:
                query = self.DocumentSummary.select().order_by(self.DocumentSummary.created_at.desc())  # 使用self.DocumentSummary
                total = query.count()
                
                summaries = query.paginate(page, page_size)
                
                return {
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'summaries': [{
                        'doc_id': s.doc_id,
                        'filename': s.filename,
                        'summary': s.summary,
                        'created_at': s.created_at,
                        'updated_at': s.updated_at
                    } for s in summaries]
                }
            except Exception as e:
                logger.error(f"分页获取文档摘要失败: {str(e)}")
                return {
                    'total': 0,
                    'page': page,
                    'page_size': page_size,
                    'summaries': []
                }

    def get_summaries_count(self) -> int:
        """获取文档摘要总数"""
        with self.db.connection_context():
            try:
                return self.DocumentSummary.select().count()  # 使用self.DocumentSummary
            except Exception as e:
                logger.error(f"获取文档摘要总数失败: {str(e)}")
                return 0
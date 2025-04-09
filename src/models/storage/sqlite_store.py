from peewee import SqliteDatabase, Model, CharField, TextField, DateTimeField
from typing import Optional, Dict, Any, List
from datetime import datetime
from src.config import settings
from src.utils import logger
import os
from .base_store import BaseDocumentStore

class SQLiteStore(BaseDocumentStore):
    def __init__(self):
        # 确保数据目录存在
        db_dir = os.path.dirname(settings.sqlite_db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        # 创建数据库连接
        db = SqliteDatabase(settings.sqlite_db_path)
        # 调用父类初始化，会自动创建模型类
        super().__init__(db)
        
        logger.info(f"成功初始化SQLite数据库: {settings.sqlite_db_path}")

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
                query = self.DocumentSummary.select().order_by(self.DocumentSummary.created_at.desc())
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
                return self.DocumentSummary.select().count()
            except Exception as e:
                logger.error(f"获取文档摘要总数失败: {str(e)}")
                return 0
from typing import Optional, Dict, Any, List

from peewee import *
from datetime import datetime
from src.config import settings
from src.utils import logger
import os

# 确保数据目录存在
db_dir = os.path.dirname(settings.sqlite_db_path)
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

db = SqliteDatabase(settings.sqlite_db_path)

class BaseModel(Model):
    class Meta:
        database = db

class DocumentSummary(BaseModel):
    doc_id = CharField(primary_key=True)
    filename = CharField()
    summary = TextField()
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'document_summaries'

class PeeweeStore:
    def __init__(self):
        try:
            db.connect()
            db.create_tables([DocumentSummary])
            logger.info(f"成功初始化Peewee数据库: {settings.sqlite_db_path}")
        except Exception as e:
            logger.error(f"初始化Peewee数据库失败: {str(e)}")
            raise

    def close(self):
        if not db.is_closed():
            db.close()

    def save_document_summary(self, doc_id: str, filename: str, summary: str) -> bool:
        try:
            DocumentSummary.insert(
                doc_id=doc_id,
                filename=filename,
                summary=summary
            ).on_conflict(
                conflict_target=[DocumentSummary.doc_id],
                update={
                    DocumentSummary.summary: summary,
                    DocumentSummary.updated_at: datetime.now()
                }
            ).execute()
            return True
        except Exception as e:
            logger.error(f"保存文档摘要失败: {str(e)}")
            return False

    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            summary = DocumentSummary.get_or_none(DocumentSummary.doc_id == doc_id)
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
        try:
            return [{
                'doc_id': s.doc_id,
                'filename': s.filename,
                'summary': s.summary,
                'created_at': s.created_at,
                'updated_at': s.updated_at
            } for s in DocumentSummary.select()]
        except Exception as e:
            logger.error(f"获取所有文档摘要失败: {str(e)}")
            return []
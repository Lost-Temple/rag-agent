from peewee import *
import os
from typing import Dict, Any, Optional, List


from src.config import settings
from src.utils import logger

# 定义Peewee数据库连接
sqlite_db = SqliteDatabase(None)

# 定义文档摘要模型
class DocumentSummary(Model):
    doc_id = CharField(primary_key=True)
    filename = CharField()
    summary = TextField()
    created_at = DateTimeField(constraints=[SQL('DEFAULT CURRENT_TIMESTAMP')])
    updated_at = DateTimeField(constraints=[SQL('DEFAULT CURRENT_TIMESTAMP')])
    
    class Meta:
        database = sqlite_db
        table_name = 'document_summaries'

class SQLiteStore:
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化SQLite存储
        
        Args:
            db_path: 数据库文件路径，如果为None则使用默认路径
        """
        # 如果没有指定数据库路径，使用配置中的路径
        if db_path is None:
            # 确保数据目录存在
            db_dir = os.path.dirname(settings.sqlite_db_path)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
            db_path = settings.sqlite_db_path
        
        self.db_path = db_path
        sqlite_db.init(db_path)
        
        self._initialize_db()
    
    def _initialize_db(self):
        """
        初始化数据库连接和表结构
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # 创建文档摘要表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_summaries (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            self.conn.commit()
            logger.info(f"成功初始化SQLite数据库: {self.db_path}")
        except Exception as e:
            logger.error(f"初始化SQLite数据库失败: {str(e)}")
            if self.conn:
                self.conn.close()
                self.conn = None
            raise
    
    def _ensure_connection(self):
        """
        确保数据库连接有效
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
    
    def close(self):
        """
        关闭数据库连接
        """
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def save_document_summary(self, doc_id: str, filename: str, summary: str) -> bool:
        """
        保存文档摘要
        
        Args:
            doc_id: 文档ID
            filename: 文件名
            summary: 文档摘要内容
            
        Returns:
            bool: 操作是否成功
        """
        try:
            self._ensure_connection()
            cursor = self.conn.cursor()
            
            # 检查是否已存在该文档的摘要
            cursor.execute("SELECT doc_id FROM document_summaries WHERE doc_id = ?", (doc_id,))
            exists = cursor.fetchone()
            
            if exists:
                # 更新现有摘要
                cursor.execute("""
                UPDATE document_summaries 
                SET summary = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE doc_id = ?
                """, (summary, doc_id))
            else:
                # 插入新摘要
                cursor.execute("""
                INSERT INTO document_summaries (doc_id, filename, summary) 
                VALUES (?, ?, ?)
                """, (doc_id, filename, summary))
            
            self.conn.commit()
            logger.info(f"成功保存文档摘要，文档ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"保存文档摘要失败: {str(e)}")
            return False
    
    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档摘要
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Dict[str, Any] or None: 文档摘要信息，如果不存在则返回None
        """
        try:
            self._ensure_connection()
            cursor = self.conn.cursor()
            
            cursor.execute("""
            SELECT doc_id, filename, summary, created_at, updated_at 
            FROM document_summaries 
            WHERE doc_id = ?
            """, (doc_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    "doc_id": result[0],
                    "filename": result[1],
                    "summary": result[2],
                    "created_at": result[3],
                    "updated_at": result[4]
                }
            return None
        except Exception as e:
            logger.error(f"获取文档摘要失败: {str(e)}")
            return None
    
    def get_all_document_summaries(self) -> List[Dict[str, Any]]:
        """
        获取所有文档摘要
        
        Returns:
            List[Dict[str, Any]]: 所有文档摘要的列表
        """
        try:
            self._ensure_connection()
            cursor = self.conn.cursor()
            
            cursor.execute("""
            SELECT doc_id, filename, summary, created_at, updated_at 
            FROM document_summaries 
            ORDER BY updated_at DESC
            """)
            
            results = cursor.fetchall()
            summaries = []
            for result in results:
                summaries.append({
                    "doc_id": result[0],
                    "filename": result[1],
                    "summary": result[2],
                    "created_at": result[3],
                    "updated_at": result[4]
                })
            return summaries
        except Exception as e:
            logger.error(f"获取所有文档摘要失败: {str(e)}")
            return []
    
    def delete_document_summary(self, doc_id: str) -> bool:
        """
        删除文档摘要
        
        Args:
            doc_id: 文档ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            self._ensure_connection()
            cursor = self.conn.cursor()
            
            cursor.execute("DELETE FROM document_summaries WHERE doc_id = ?", (doc_id,))
            self.conn.commit()
            
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"删除文档摘要失败: {str(e)}")
            return False
from typing import List, Dict, Any
from neo4j import GraphDatabase
from src.config import settings

class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
    
    def create_document_node(self, doc_id: str, metadata: Dict[str, Any]):
        """创建文档节点"""
        with self.driver.session() as session:
            session.execute_write(self._create_document_node, doc_id, metadata)
    
    def _create_document_node(self, tx, doc_id: str, metadata: Dict[str, Any]):
        query = """
        MERGE (d:Document {doc_id: $doc_id})
        SET d += $metadata
        """
        tx.run(query, doc_id=doc_id, metadata=metadata)
    
    def create_chunk_node(self, chunk_id: str, doc_id: str, content: str, metadata: Dict[str, Any]):
        """创建文档片段节点并与文档建立关系"""
        with self.driver.session() as session:
            session.execute_write(
                self._create_chunk_node,
                chunk_id,
                doc_id,
                content,
                metadata
            )
    
    def _create_chunk_node(self, tx, chunk_id: str, doc_id: str, content: str, metadata: Dict[str, Any]):
        query = """
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.content = $content
        SET c += $metadata
        MERGE (d)-[:CONTAINS]->(c)
        """
        tx.run(
            query,
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=content,
            metadata=metadata
        )
    
    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """获取文档元数据"""
        with self.driver.session() as session:
            result = session.execute_read(self._get_document_metadata, doc_id)
            return result
    
    def _get_document_metadata(self, tx, doc_id: str):
        query = """
        MATCH (d:Document {doc_id: $doc_id})
        RETURN d
        """
        result = tx.run(query, doc_id=doc_id)
        record = result.single()
        return record["d"] if record else None
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """获取文档的所有片段"""
        with self.driver.session() as session:
            result = session.execute_read(self._get_document_chunks, doc_id)
            return result
    
    def _get_document_chunks(self, tx, doc_id: str):
        query = """
        MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
        RETURN c
        """
        result = tx.run(query, doc_id=doc_id)
        return [record["c"] for record in result]
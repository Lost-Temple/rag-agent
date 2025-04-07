from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, Any
import uuid
import os
from src.models.document_processor import DocumentProcessor
from src.models import Vectorizer, GraphStore
from src.config import settings
from src.api.llm_service import router as llm_router

app = FastAPI(title="RAG System API")

# 添加LLM路由
app.include_router(llm_router, prefix="/llm", tags=["LLM"])

class RAGSystem:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vectorizer = Vectorizer()
        self.graph_store = GraphStore()
        
        # 确保向量存储目录存在
        if not os.path.exists(settings.vector_store_path):
            os.makedirs(settings.vector_store_path)
        else:
            self.vectorizer.load_vector_store()

rag_system = RAGSystem()

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        # 生成唯一文档ID
        doc_id = str(uuid.uuid4())
        
        # 确保原始文档存储目录存在
        if not os.path.exists(settings.original_documents_path):
            os.makedirs(settings.original_documents_path)
            
        # 保存上传的文件到原始文档存储目录
        original_file_path = os.path.join(settings.original_documents_path, f"{doc_id}_{file.filename}")
        temp_file_path = f"temp_{file.filename}"
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
                
            # 同时保存一份到原始文档存储目录
            with open(original_file_path, "wb") as original_file:
                original_file.write(content)
            
            # 处理文档
            documents = rag_system.doc_processor.process_document(temp_file_path)
            
            # 提取元数据
            metadata = {
                "doc_id": doc_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "original_file_path": original_file_path
            }
            
            # 存储到图数据库
            rag_system.graph_store.create_document_node(doc_id, metadata)
            
            # 为每个文档片段创建节点
            for i, doc in enumerate(documents):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({  # chunk的元数据
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": i
                })
                
                rag_system.graph_store.create_chunk_node(
                    chunk_id,
                    doc_id,
                    doc.page_content,
                    chunk_metadata
                )
            
            # 更新向量存储
            rag_system.vectorizer.initialize_vector_store(documents)
            
            # 生成文档摘要
            summary = await rag_system.doc_processor.generate_document_summary(
                original_file_path,
                doc_id,
                file.filename
            )
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "message": "Document processed and stored successfully",
                "chunks_count": len(documents),
                "summary_generated": summary is not None
            }
            
        finally:
            # 只清理临时文件，保留原始文档
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
def get_document(doc_id: str) -> Dict[str, Any]:
    try:
        metadata = rag_system.graph_store.get_document_metadata(doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        chunks = rag_system.graph_store.get_document_chunks(doc_id)
        
        # 获取文档摘要（如果存在）
        summary_info = rag_system.doc_processor.get_document_summary(doc_id)
        
        return {
            "doc_id": doc_id,
            "metadata": metadata,
            "chunks": chunks,
            "summary": summary_info["summary"] if summary_info else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/summaries/all")
def get_all_document_summaries() -> Dict[str, Any]:
    try:
        # 获取所有文档摘要
        summaries = rag_system.doc_processor.get_all_document_summaries()
        
        return {
            "status": "success",
            "count": len(summaries),
            "summaries": summaries
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port
    )
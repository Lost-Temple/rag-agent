from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from typing import Dict, Any, Optional
import uuid
import os

from pydantic import BaseModel

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
        
        # 创建以doc_id为名的子目录
        doc_dir = os.path.join(settings.original_documents_path, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
            
        # 保存上传的文件到文档子目录，使用原始文件名
        original_file_path = os.path.join(doc_dir, file.filename)
        try:
            with open(original_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # 处理文档
            documents = rag_system.doc_processor.process_document(original_file_path)

            # 提取元数据
            metadata = {
                "doc_id": doc_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "original_file_path": original_file_path,
                "doc_dir": doc_dir  # 添加文档目录路径到元数据
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
                    chunk_id, # chunk的ID
                    doc_id,  # 所属文档的ID
                    doc.page_content,  # 文档片段的内容
                    chunk_metadata  # chunk的元数据可以自定义，保存到图数据库中
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
            # 不再需要清理临时文件
            pass
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
def get_document(doc_id: str) -> Dict[str, Any]:
    try:
        metadata = rag_system.graph_store.get_document_metadata(doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # 检查文档目录是否存在
        doc_dir = metadata.get("doc_dir")
        if not doc_dir or not os.path.exists(doc_dir):
            raise HTTPException(status_code=404, detail="Document directory not found")
        
        chunks = rag_system.graph_store.get_document_chunks(doc_id)
        
        # 获取文档摘要（如果存在）
        summary_info = rag_system.doc_processor.get_document_summary(doc_id)

        # 确保metadata和chunks都是可序列化的字典类型
        serializable_metadata = dict(metadata) if metadata else {}
        serializable_chunks = [dict(chunk) for chunk in chunks] if chunks else []
        # 要移除的键
        keys_to_remove = ['source', 'doc_id']

        # 创建一个新的列表，其中每个字典都已移除指定的键
        serializable_chunks = [
            {k: v for k, v in chunk.items() if k not in keys_to_remove}
            for chunk in serializable_chunks
        ]

        return {
            "doc_id": doc_id,
            "metadata": serializable_metadata,
            "chunks": serializable_chunks,
            "summary": summary_info["summary"] if summary_info else None
        }
    
    except HTTPException:
        # 直接重新抛出HTTP异常，保持原始状态码
        raise
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


from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# 添加MCP相关的路由
mcp_router = APIRouter(prefix="/mcp", tags=["MCP工具"])

class MCPQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    tool_name: str

class MCPResponse(BaseModel):
    status: str
    result: str

@mcp_router.post("/tool", response_model=MCPResponse)
async def call_mcp_tool(request: MCPQueryRequest):
    """
    调用MCP工具执行查询
    
    Args:
        request: 包含查询文本、top_k参数和工具名称的请求
    
    Returns:
        查询结果
    """
    try:
        from src.mcp.mcp_client import MCPClient
        
        # 创建MCP客户端
        client = MCPClient()
        
        # 根据工具名称调用相应的工具
        if request.tool_name == "hybrid_search":
            result = await client.call_hybrid_search(request.query, request.top_k)
        elif request.tool_name == "generate_answer":
            result = await client.call_generate_answer(request.query, request.top_k)
        else:
            raise HTTPException(status_code=400, detail=f"未知的工具名称: {request.tool_name}")
        
        return {"status": "success", "result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 将MCP路由器添加到主应用
app.include_router(mcp_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port
    )
import argparse
import uvicorn
import sys
import os
from src.models.llm.ollama_llm import OllamaLLMClient
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server

from src.models import Vectorizer
from src.config import settings

# 设置日志
from src.utils import logger

# 初始化FastMCP服务器，用于RAG检索
mcp = FastMCP("rag_retrieval")

class RetrievalService:
    def __init__(self):
        self.vectorizer = Vectorizer()
        self.vectorizer.load_vector_store()
        # 不在这里调用load_vector_store，而是在使用前检查并加载
        self.llm = OllamaLLMClient()
    
    def is_initialized(self):
        """检查向量库是否已初始化"""
        return hasattr(self.vectorizer, 'vector_store') and self.vectorizer.vector_store is not None

# 创建检索服务实例
retrieval_service = RetrievalService()

@mcp.tool()
async def hybrid_search(query: str, top_k: int = 5) -> str:
    """
    执行混合检索，结合向量检索和重排序，返回最相关的文档。

    Args:
        query: 用户的查询问题
        top_k: 返回的结果数量
    """
    logger.info(f"调用了MCP工具: hybrid_search, query: {query}, top_k: {top_k}")
    try:
        # 确保向量库已加载
        if not retrieval_service.is_initialized():
            logger.info("混合检索, 向量库未初始化, 加载向量库...")
            retrieval_service.vectorizer.load_vector_store()
            
        # 执行混合检索
        results = retrieval_service.vectorizer.hybrid_search(
            query=query,
            k=top_k
        )
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results):
            formatted_result = f"""\n文档 {i+1}:\n"""
            formatted_result += f"内容: {result['content']}\n"
            formatted_result += f"相关度分数: {result['final_score']:.4f}\n"
            
            # 添加元数据信息
            if result['metadata']:
                formatted_result += "元数据:\n"
                for key, value in result['metadata'].items():
                    formatted_result += f"  {key}: {value}\n"
            
            formatted_results.append(formatted_result)
        
        return "\n---\n".join(formatted_results)
    
    except Exception as e:
        logger.error(f"混合搜索错误: {str(e)}")
        return f"检索过程中发生错误: {str(e)}"

@mcp.tool()
async def generate_answer(query: str, top_k: int = 5) -> str:
    """
    基于用户查询生成回答，使用RAG方法检索相关文档并生成回答。

    Args:
        query: 用户的查询问题
        top_k: 检索的文档数量
    """
    logger.info(f"调用了MCP工具: generate_answer, query: {query}, top_k: {top_k}")
    try:
        # 确保向量库已加载
        if not retrieval_service.is_initialized():
            logger.info("生成回答, 向量库未初始化, 加载向量库...")
            retrieval_service.vectorizer.load_vector_store()
            
        # 执行混合检索获取上下文
        context = retrieval_service.vectorizer.hybrid_search(
            query=query,
            k=top_k
        )
        
        # 使用LLM生成回答
        if retrieval_service.llm.use_ollama:
            # 修复：添加await关键字等待协程执行完成
            answer = await retrieval_service.llm.generate_answer(query, context)
            logger.info(f"LLM生成回答: {answer}")
            return answer
        else:
            # 如果LLM未启用，返回检索结果
            formatted_results = []
            for i, result in enumerate(context):
                formatted_results.append(f"文档 {i+1}:\n{result['content']}")
            
            return "LLM未启用，仅返回检索结果:\n\n" + "\n\n".join(formatted_results)
    
    except Exception as e:
        logger.error(f"生成回答错误: {str(e)}")
        return f"生成回答过程中发生错误: {str(e)}"

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """创建一个Starlette应用，用于提供MCP服务器的SSE服务。"""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

def serve():
    """启动MCP服务器"""
    mcp_server = mcp._mcp_server  # noqa: WPS437
    
    parser = argparse.ArgumentParser(description='运行基于SSE的MCP RAG检索服务')
    parser.add_argument('--host', default=settings.mcp_host, help='绑定的主机地址')
    parser.add_argument('--port', type=int, default=settings.mcp_port, help='监听的端口')
    args = parser.parse_args()

    # 将SSE请求处理绑定到MCP服务器
    starlette_app = create_starlette_app(mcp_server, debug=True)

    print(f"MCP RAG检索服务已启动，地址: {args.host}:{args.port}")
    uvicorn.run(starlette_app, host=args.host, port=args.port)

if __name__ == "__main__":
    serve()
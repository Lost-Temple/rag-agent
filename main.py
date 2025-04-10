import asyncio
import signal
import uvicorn
from src.api.api_service import app
from src.config import settings
from starlette.requests import Request
from mcp.server.sse import SseServerTransport
from src.utils import logger

class ServiceManager:
    def __init__(self):
        self._running = True
        self._api_server = None
        self._event_loop = None
        self._resources_initialized = False
        self._rag_system = None
    
    def _setup_signal_handlers(self):
        # 设置信号处理器用于优雅关闭
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        logger.info("Received shutdown signal. Initiating graceful shutdown...")
        self._running = False
        # 调用shutdown方法来优雅地关闭所有资源
        self.shutdown()
    
    def shutdown(self):
        """优雅地关闭所有服务和资源"""
        try:
            # 停止API服务器
            if self._api_server:
                logger.info("Stopping API server...")
                self._api_server.should_exit = True
            
            # 关闭RAG系统资源
            if self._rag_system:
                logger.info("Closing RAG system resources...")
                try:
                    # 关闭图数据库连接
                    if hasattr(self._rag_system, 'graph_store'):
                        logger.info("Closing graph database connection...")
                        self._rag_system.graph_store.close()
                    
                    # 关闭数据库连接
                    if hasattr(self._rag_system, 'doc_processor') and hasattr(self._rag_system.doc_processor, 'store'):
                        logger.info("Closing database connection...")
                        try:
                            self._rag_system.doc_processor.store.close()
                        except Exception as e:
                            logger.error(f"Error closing database connection: {str(e)}")
                except Exception as e:
                    logger.error(f"Error closing database connections: {str(e)}")
            
            # 优雅地关闭事件循环：取消所有未完成的任务
            if self._event_loop:
                logger.info("Cancelling all pending tasks...")
                # 获取所有未完成的任务
                pending_tasks = [task for task in asyncio.all_tasks(self._event_loop) 
                               if not task.done() and task is not asyncio.current_task()]
                
                if pending_tasks:
                    # 取消所有未完成的任务
                    for task in pending_tasks:
                        task.cancel()
                    
                    # 等待所有任务完成取消操作
                    logger.info(f"Waiting for {len(pending_tasks)} tasks to cancel...")
                    try:
                        # 使用gather和短暂超时等待任务取消
                        self._event_loop.run_until_complete(
                            asyncio.gather(*pending_tasks, return_exceptions=True)
                        )
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error while cancelling tasks: {str(e)}")
                
                # 最后停止事件循环
                logger.info("Stopping event loop...")
                self._event_loop.stop()
                
            logger.info("All resources have been gracefully shut down.")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            # 即使出现错误，也要尝试停止事件循环
            if self._event_loop:
                # 尝试取消任务
                try:
                    pending_tasks = [task for task in asyncio.all_tasks(self._event_loop) 
                                   if not task.done() and task is not asyncio.current_task()]
                    for task in pending_tasks:
                        task.cancel()
                except Exception:
                    pass
                self._event_loop.stop()
    
    def _setup_mcp_server(self):
        """设置MCP服务器并将其挂载到FastAPI应用上"""
        # 导入MCP服务模块
        from src.mcp.mcp_service import mcp
        
        # 获取MCP服务器实例
        mcp_server = mcp._mcp_server
        
        # 创建SSE传输
        sse = SseServerTransport("/mcp/messages/")
        
        # 定义SSE处理函数
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
        
        # 将MCP的SSE端点挂载到FastAPI应用
        from fastapi import APIRouter
        mcp_sse_router = APIRouter()
        
        # 添加SSE路由
        from starlette.routing import Route
        # 注意：这里不需要前缀，因为我们会在include_router时添加前缀
        mcp_sse_router.routes.append(Route("/sse", endpoint=handle_sse))
        
        # 挂载消息处理路由
        app.mount("/mcp/messages", sse.handle_post_message)
        
        # 将SSE路由添加到主应用，使用前缀
        app.include_router(mcp_sse_router, prefix="/mcp")
        
        logger.info("MCP SSE服务已挂载到API服务器")
    
    async def _start_combined_server(self):
        """启动合并后的服务器"""
        # 设置MCP服务器
        self._setup_mcp_server()
        
        # 配置并启动合并后的服务器
        config = uvicorn.Config(
            app=app,
            host=settings.api_host,
            port=settings.api_port,
            loop="asyncio"
        )
        self._api_server = uvicorn.Server(config)
        logger.info(f"Combined API+MCP Server starting on {settings.api_host}:{settings.api_port}")
        await self._api_server.serve()
    
    async def start_all_services(self):
        logger.info("Starting all services...")
        self._setup_signal_handlers()
        self._event_loop = asyncio.get_event_loop()
        
        # 预加载LLM服务
        logger.info("Initializing LLM service...")
        # LLM服务作为API服务的一部分，通过路由集成，不需要单独启动
        # 但我们可以预先导入以确保它被初始化
        import src.api.llm_service
        
        # 获取RAG系统实例，用于在关闭时释放资源
        from src.api.api_service import rag_system
        self._rag_system = rag_system
        
        # 初始化向量库
        logger.info("Initializing vector store...")
        if hasattr(self._rag_system, 'vectorizer'):
            try:
                # 加载向量库
                self._rag_system.vectorizer.load_vector_store()
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load vector store: {str(e)}")
                # 如果向量库加载失败，可以考虑是否要继续启动服务
                # 这里选择继续启动，但会记录错误
        else:
            logger.warning("RAG system does not have a vectorizer attribute")
        
        self._resources_initialized = True
        
        # 启动合并后的服务器
        logger.info("Starting combined API+MCP server...")
        await self._start_combined_server()

def main():
    service_manager = ServiceManager()
    try:
        asyncio.run(service_manager.start_all_services())
    except KeyboardInterrupt:
        # KeyboardInterrupt已经通过信号处理器处理
        pass
    except Exception as e:
        logger.error(f"程序异常退出: {str(e)}")
    finally:
        # 确保在任何情况下都能完成优雅退出
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
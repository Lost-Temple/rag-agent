import asyncio
import signal
import uvicorn
from src.api.api_service import app
from src.config import settings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from src.utils import logger

class ServiceManager:
    def __init__(self):
        self._running = True
        self._api_server = None
        self._mcp_server = None
        self._mcp_app = None
        self._event_loop = None
    
    def _setup_signal_handlers(self):
        # 设置信号处理器用于优雅关闭
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        logger.info("Received shutdown signal. Initiating graceful shutdown...")
        self._running = False
        if self._api_server:
            logger.info("Stopping API server...")
            self._api_server.should_exit = True
        if self._mcp_server:
            logger.info("Stopping MCP server...")
            # MCP服务器没有明确的关闭方法，依赖于事件循环的停止
        if self._event_loop:
            self._event_loop.stop()
    
    async def _start_api_server(self):
        config = uvicorn.Config(
            app=app,
            host=settings.api_host,
            port=settings.api_port,
            loop="asyncio"
        )
        self._api_server = uvicorn.Server(config)
        logger.info(f"API Server starting on {settings.api_host}:{settings.api_port}")
        await self._api_server.serve()
        
    def _create_starlette_app(self, mcp_server: Server, *, debug: bool = False) -> Starlette:
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
        
    async def _start_mcp_server(self):
        # 导入MCP服务模块
        from src.mcp.mcp_service import mcp
        
        # 获取MCP服务器实例
        mcp_server = mcp._mcp_server
        self._mcp_server = mcp_server
        
        # 创建Starlette应用
        starlette_app = self._create_starlette_app(mcp_server, debug=True)
        self._mcp_app = starlette_app
        
        # 配置并启动MCP服务器
        config = uvicorn.Config(
            app=starlette_app,
            host=settings.mcp_host,
            port=settings.mcp_port,
            loop="asyncio"
        )
        mcp_uvicorn = uvicorn.Server(config)
        logger.info(f"MCP Server starting on {settings.mcp_host}:{settings.mcp_port}")
        await mcp_uvicorn.serve()
    
    async def start_all_services(self):
        logger.info("Starting all services...")
        self._setup_signal_handlers()
        self._event_loop = asyncio.get_event_loop()
        
        # 预加载LLM服务
        logger.info("Initializing LLM service...")
        # LLM服务作为API服务的一部分，通过路由集成，不需要单独启动
        # 但我们可以预先导入以确保它被初始化
        import src.api.llm_service
        
        # 并行启动所有服务
        logger.info("Starting all services in parallel...")
        await asyncio.gather(
            self._start_api_server(),
            self._start_mcp_server()
        )

def main():
    service_manager = ServiceManager()
    try:
        asyncio.run(service_manager.start_all_services())
    except KeyboardInterrupt:
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
import asyncio
from contextlib import AsyncExitStack
import json
from mcp import ClientSession
from mcp.client.sse import sse_client
import os
import sys
from typing import Optional, Dict, Any

from src.config import settings
from src.utils import logger

class MCPClient:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        """
        初始化MCP客户端
        
        Args:
            host: MCP服务器主机地址，默认使用配置中的地址
            port: MCP服务器端口，默认使用配置中的端口
        """
        # 设置服务器地址
        self.host = host or settings.api_host
        self.port = port or settings.api_port
        # 更新URL路径，使用合并后的MCP路径
        self.server_url = f"http://localhost:{self.port}/mcp/sse"
        
        # 初始化会话和上下文管理
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._streams_context = None
        self._session_context = None
        self._connected = False

    async def connect(self):
        """连接到MCP服务器"""
        if self._connected:
            return
            
        try:
            logger.info(f"正在连接MCP服务器: {self.server_url}")
            # 创建SSE客户端连接，使用正确的URL
            self._streams_context = sse_client(url=self.server_url)
            streams = await self._streams_context.__aenter__()

            # 创建会话
            self._session_context = ClientSession(*streams)
            self.session: ClientSession = await self._session_context.__aenter__()

            # 初始化会话
            await self.session.initialize()
            self._connected = True
            
            # 获取可用工具列表
            response = await self.session.list_tools()
            tools = response.tools
            logger.info(f"已连接到MCP服务器，可用工具: {[tool.name for tool in tools]}")
            
        except Exception as e:
            # 连接失败时清理资源
            await self.disconnect()
            logger.error(f"连接MCP服务器失败: {str(e)}")
            raise Exception(f"连接MCP服务器失败: {str(e)}")

    async def disconnect(self):
        """断开与MCP服务器的连接"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
            self._session_context = None
            
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)
            self._streams_context = None
            
        self._connected = False

    async def call_tool(self, tool_name: str, **params) -> Any:
        """
        调用MCP工具
        
        Args:
            tool_name: 工具名称
            params: 工具参数
            
        Returns:
            工具执行结果
        """
        try:
            # 确保已连接
            if not self._connected:
                await self.connect()
                
            # 调用工具
            result = await self.session.call_tool(tool_name, params)
            
            # 处理不同类型的响应
            if hasattr(result, 'content'):
                content = result.content
                # 检查是否是协程对象的字符串表示
                if isinstance(content, list) and len(content) > 0:
                    # 处理列表类型的内容
                    text_contents = []
                    for item in content:
                        if hasattr(item, 'text'):
                            text = item.text
                            # 检查是否是协程对象的字符串表示
                            if text and text.startswith('<coroutine object'):
                                # 这里需要服务端修复，客户端无法直接执行协程
                                print(f"警告: 服务器返回了协程对象: {text}")
                                # 尝试再次调用工具，使用不同的参数
                                if tool_name == "generate_answer":
                                    # 对于生成回答，尝试直接调用服务器上的LLM
                                    return await self.call_tool("direct_llm_call", 
                                                               query=params.get('query', ''),
                                                               context=f"查询: {params.get('query', '')}")
                            elif text:  # 只添加非空文本
                                text_contents.append(text)
                    
                    if text_contents:
                        return "\n".join(text_contents)
                    else:
                        # 如果没有有效内容，尝试直接调用LLM
                        if tool_name == "generate_answer":
                            print("尝试直接调用LLM...")
                            return await self.call_tool("direct_llm_call", 
                                                      query=params.get('query', ''),
                                                      context="")
                        return "服务器返回了空结果"
                else:
                    # 其他情况，直接返回内容
                    return str(content)
            else:
                # 如果没有content属性，返回整个结果的字符串表示
                return str(result)
            
        except Exception as e:
            raise Exception(f"调用工具 {tool_name} 失败: {str(e)}")

    async def call_hybrid_search(self, query: str, top_k: int = 5) -> str:
        """
        调用混合搜索工具
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果
        """
        try:
            result = await self.call_tool("hybrid_search", query=query, top_k=top_k)
            return result
        finally:
            # 调用完成后断开连接
            await self.disconnect()
    
    async def call_generate_answer(self, query: str, top_k: int = 5) -> str:
        """
        调用生成回答工具
        
        Args:
            query: 查询文本
            top_k: 检索文档数量
            
        Returns:
            生成的回答
        """
        try:
            result = await self.call_tool("generate_answer", query=query, top_k=top_k)
            return result
        finally:
            # 调用完成后断开连接
            await self.disconnect()

# 测试代码
async def test_client():
    client = MCPClient()
    try:
        print("测试混合搜索...")
        result = await client.call_hybrid_search("什么是向量数据库?", 3)
        print("混合搜索结果:", result)
        
        print("\n测试生成回答...")
        answer = await client.call_generate_answer("解释一下知识图谱的作用", 3)
        print("生成回答:", answer)
    except Exception as e:
        print("客户端错误:", str(e))

if __name__ == "__main__":
    asyncio.run(test_client())
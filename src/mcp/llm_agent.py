import asyncio
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from src.config import settings
from src.mcp.mcp_client import MCPClient
from src.utils import logger

class LLMAgent:
    """
    基于大语言模型的智能代理，使用LlamaIndex ReActAgent实现
    """

    def __init__(self):
        self.use_ollama = None
        self.llm = None
        self.agent = None
        self.mcp_client = None

    async def initialize(self):
        """异步初始化方法"""
        self.use_ollama = settings.use_ollama

        # 初始化LLM
        if self.use_ollama:
            try:
                self.llm = Ollama(
                    model=settings.ollama_model,
                    base_url=settings.ollama_base_url,
                    temperature=0.3,
                    context_window=8192
                )
                logger.info(f"成功初始化Ollama LLM，模型: {settings.ollama_model}")
            except Exception as e:
                logger.error(f"初始化Ollama LLM失败: {str(e)}")
                self.llm = None
        else:
            logger.error("目前不支持Ollama之外的LLM提供商")
            self.llm = None

        # 检查LLM是否初始化成功
        if self.llm is None:
            raise RuntimeError("LLM初始化失败，请检查配置")

        # 获取并转换工具
        self.mcp_client = MCPClient()
        await self.mcp_client.connect()
        
        # 获取工具列表
        mcp_tools = (await self.mcp_client.session.list_tools()).tools
        
        # 将MCP工具转换为LlamaIndex FunctionTool对象
        tools = []
        for tool in mcp_tools:
            # 创建同步包装函数，确保正确捕获工具名称
            def create_sync_tool(current_tool_name, current_tool_description):
                async def async_tool_caller(query: str, **kwargs):
                    try:
                        logger.info(f"调用工具: {current_tool_name}, 查询: {query}, 参数: {kwargs}")
                        # 添加工具调用超时
                        try:
                            result = await asyncio.wait_for(
                                self.mcp_client.call_tool(current_tool_name, query=query, **kwargs),
                                timeout=120.0  # 增加到120秒
                            )
                            logger.info(f"工具 {current_tool_name} 返回结果: {result[:100] if isinstance(result, str) else str(result)[:100]}...")
                            return result
                        except asyncio.TimeoutError:
                            logger.error(f"工具 {current_tool_name} 调用超时")
                            return f"工具 {current_tool_name} 调用超时，请稍后再试"
                    except Exception as e:
                        logger.error(f"调用工具 {current_tool_name} 失败: {str(e)}")
                        return f"工具调用失败: {str(e)}"
                
                def sync_func(query: str, **kwargs):
                    """同步包装函数，在新事件循环中执行异步调用"""
                    logger.info(f"开始同步调用工具: {current_tool_name}, 参数: {kwargs}")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(async_tool_caller(query, **kwargs))
                        logger.info(f"同步调用工具 {current_tool_name} 完成")
                        return result
                    except Exception as e:
                        logger.error(f"同步调用工具 {current_tool_name} 异常: {str(e)}")
                        return f"工具调用失败: {str(e)}"
                    finally:
                        loop.close()
                
                return FunctionTool.from_defaults(
                    name=current_tool_name,
                    description=current_tool_description or f"{current_tool_name} 工具",
                    fn=sync_func
                )
            
            tools.append(create_sync_tool(tool.name, tool.description))
        
        logger.info(f"已加载 {len(tools)} 个工具")
        
        # 初始化ReActAgent
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            verbose=True,
            max_iterations=10
        )

    async def invoke(self, question: str) -> str:
        """执行查询并返回结果"""
        logger.info(f"处理问题: {question}")
        
        # 添加重试机制
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # 在执行器中运行同步代理，添加超时
                loop = asyncio.get_event_loop()
                response_future = loop.run_in_executor(
                    None, lambda: self.agent.query(question)
                )
                
                # 添加超时处理
                try:
                    response = await asyncio.wait_for(response_future, timeout=180.0)  # 增加到180秒
                except asyncio.TimeoutError:
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(f"代理查询超时，正在进行第 {retry_count} 次重试...")
                        continue
                    logger.error("代理查询超时，已达到最大重试次数")
                    return "处理您的问题超时，请稍后再试或尝试简化您的问题。如果您询问的是复杂问题，系统可能需要更长时间处理。"
                
                # 提取响应文本
                if hasattr(response, 'response'):
                    result = response.response
                else:
                    result = str(response)
                    
                logger.info(f"生成回答成功: {result[:100]}...")
                return result
                
            except Exception as e:
                retry_count += 1
                logger.error(f"调用代理时发生错误: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                if retry_count <= max_retries:
                    logger.warning(f"发生错误，正在进行第 {retry_count} 次重试...")
                    continue
                
                return f"调用代理时发生错误: {str(e)}"

# 测试代码
async def test_agent():
    agent = LLMAgent()
    await agent.initialize()
    try:
        print("测试简单问题...")
        result = await agent.invoke("什么是向量数据库?")
        print("回答:", result)
    except Exception as e:
        print("代理错误:", str(e))

if __name__ == "__main__":
    asyncio.run(test_agent())
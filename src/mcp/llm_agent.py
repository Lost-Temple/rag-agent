import asyncio
from langchain.agents import initialize_agent, AgentType
from src.config import settings
from src.mcp.mcp_client import MCPClient
from src.models.llm.ollama_llm import OllamaLLMClient
from src.utils import logger

class LLMAgent:
    """
    基于大语言模型的智能代理，封装LangChain Agent功能
    """

    def __init__(self):
        self.use_ollama = None
        self.llm = None
        self.agent = None

    async def initialize(self):
        """异步初始化方法"""
        self.use_ollama = settings.use_ollama

        if self.use_ollama:
            self.llm = OllamaLLMClient().llm
        else:
            logger.error("目前不支持Ollama之外的LLM提供商")
            self.llm = None  # 确保llm属性始终存在

        # 检查LLM是否初始化成功
        if self.llm is None:
            raise RuntimeError("LLM初始化失败，请检查配置")

        # 获取并转换工具
        self.mcp_client = MCPClient()
        await self.mcp_client.connect()
        
        # 获取工具列表
        mcp_tools = (await self.mcp_client.session.list_tools()).tools
        
        # 将MCP工具转换为LangChain Tool对象
        from langchain.agents import Tool
        tools = []
        for tool in mcp_tools:
            def create_tool_func(tool_name):
                async def async_func(query):
                    try:
                        return await self.mcp_client.call_tool(tool_name, query=query)
                    except Exception as e:
                        logger.error(f"调用工具{tool_name}失败: {str(e)}")
                        return f"工具调用失败: {str(e)}"
                
                def sync_func(query):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(async_func(query))
                    finally:
                        loop.close()
                
                return sync_func
                
            tools.append(Tool(
                name=tool.name,
                func=create_tool_func(tool.name),
                description=tool.description or f"{tool.name} tool"
            ))
        logger.info(f"工具列表: {tools}")
        # 初始化代理
        self.agent = initialize_agent(tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    async def invoke(self, question):
        """执行查询并返回结果"""
        logger.info(f"将会调用远程工具")
        try:
            # 使用同步方法invoke代替异步方法ainvoke
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.agent.invoke(question))
            
            # 检查结果
            logger.info(f"代理返回结果类型: {type(result)}")
            if hasattr(result, 'return_values') and 'output' in result.return_values:
                return result.return_values['output']
            
            return result
            
        except Exception as e:
            logger.error(f"调用代理时发生错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"调用代理时发生错误: {str(e)}"

# 测试代码
async def test_agent():
    agent = LLMAgent()
    try:
        # 测试简单问题
        print("测试简单问题...")
        result = await agent.invoke("什么是向量数据库?")
        print("回答:", result)
    except Exception as e:
        print("代理错误:", str(e))

if __name__ == "__main__":
    asyncio.run(test_agent())
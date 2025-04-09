import asyncio
from src.mcp.llm_agent import LLMAgent  # 修改导入路径

async def main():
    agent = LLMAgent()
    await agent.initialize()
    print("LLM智能代理已启动(输入'退出'结束)")
    
    while True:
        question = input("\n请输入您的问题: ")
        if question.lower() in ['退出', 'exit', 'quit']:
            break
        
        print("正在处理您的问题，请稍候...")
        try:
            response = await agent.invoke(question)
            print(f"\n回答: {response}")
        except Exception as e:
            print(f"\n处理问题时出错: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
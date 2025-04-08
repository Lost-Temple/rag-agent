import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mcp.mcp_client import MCPClient

async def test_hybrid_search():
    client = MCPClient()
    try:
        print("测试混合搜索...")
        query = "票据系统的功能需求有什么?"
        result = await client.call_hybrid_search(query, top_k=3)
        print(f"查询: {query}")
        print("结果:")
        print(result)
        return True
    except Exception as e:
        print(f"混合搜索测试失败: {str(e)}")
        return False

async def test_generate_answer():
    client = MCPClient()
    try:
        print("\n测试生成回答...")
        query = "票据系统的功能需求有什么"
        result = await client.call_generate_answer(query, top_k=3)
        print(f"问题: {query}")
        print("回答:")
        print(result)
        return True
    except Exception as e:
        print(f"生成回答测试失败: {str(e)}")
        return False

async def run_tests():
    search_success = await test_hybrid_search()
    answer_success = await test_generate_answer()
    
    if search_success and answer_success:
        print("\n所有测试通过!")
    else:
        print("\n部分测试失败")

if __name__ == "__main__":
    asyncio.run(run_tests())
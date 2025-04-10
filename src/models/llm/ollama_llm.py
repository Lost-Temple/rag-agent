from typing import List, Dict, Any
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from src.config import settings

# 定义OllamaLLM的配置参数
OLLAMA_CONFIG = {
    "base_url": settings.ollama_base_url,  # Ollama服务的基础URL
    "model": settings.ollama_model,        # 使用的模型名称
    "num_ctx": 8192,                      # 上下文窗口大小
    "temperature": 0.3                    # 温度参数
}

# 定义嵌入模型的配置参数
EMBEDDING_CONFIG = {
    "base_url": settings.ollama_base_url,
    "model": settings.ollama_embedding_model
}

class OllamaLLMClient:
    def __init__(self):
        self.use_ollama = settings.use_ollama

        if self.use_ollama:
            # 初始化Ollama LLM
            self.llm = OllamaLLM(**OLLAMA_CONFIG)

            # 初始化Ollama嵌入模型（如果需要使用Ollama进行嵌入）
            self.embedding_model = OllamaEmbeddings(**EMBEDDING_CONFIG)

            # 初始化RAG提示模板
            self.rag_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""使用以下检索到的上下文信息来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。

上下文信息:
{context}

问题: {question}

回答:"""
            )

            # 初始化RAG链（使用RunnableSequence替代LLMChain）
            self.rag_chain = self.rag_prompt | self.llm

    def get_embedding_model(self):
        """获取嵌入模型（如果使用Ollama）"""
        if self.use_ollama:
            return self.embedding_model
        return None

    async def generate_answer(self, question: str, context: List[Dict[str, Any]]) -> str:
        """根据检索到的上下文生成回答"""
        if not self.use_ollama:
            return "Ollama LLM未启用，请在配置中启用。"

        # 格式化上下文
        formatted_context = "\n\n".join([f"文档 {i + 1}:\n{doc['content']}" for i, doc in enumerate(context)])

        # 生成回答
        try:
            response = await self.rag_chain.ainvoke({
                "context": formatted_context,
                "question": question
            })
            return response
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

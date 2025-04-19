from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.models import Vectorizer
from src.models.llm.ollama_llm import OllamaLLMClient

router = APIRouter()

# 初始化LLM和向量化器
llm = OllamaLLMClient()
vectorizer = Vectorizer()
vectorizer.load_vector_store()


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class AnswerResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]


@router.post("/question", response_model=AnswerResponse, operation_id="answer_question",
             description="根据用户输入进行混合查询，返回重排打分后的查询结果")
async def answer_question(request: QuestionRequest):
    try:
        # 检索相关文档
        context = vectorizer.hybrid_search(request.question, k=request.top_k)

        # 使用LLM生成回答
        answer = await llm.generate_answer(request.question, context)

        return {
            "answer": answer,
            "context": context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

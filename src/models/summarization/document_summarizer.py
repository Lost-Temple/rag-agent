from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from src.models.llm.ollama_llm import OllamaLLMClient
from src.config import settings
from src.utils import logger
from langchain_core.prompts import PromptTemplate

class DocumentSummarizer:
    def __init__(self):
        """
        初始化文档摘要生成器
        """
        self.llm_client = OllamaLLMClient()
        self.llm = self.llm_client.llm
        
        # 初始化摘要提示模板
        self.summary_prompt = PromptTemplate(
            input_variables=["content"],
            template="""请对以下文档内容进行摘要总结，提取文档的主要内容、关键点和重要信息。
            生成一个全面但简洁的摘要，确保包含文档的核心内容。

文档内容:
{content}

摘要:"""
        )
        
        # 初始化合并摘要提示模板
        self.merge_prompt = PromptTemplate(
            input_variables=["summaries"],
            template="""以下是同一文档不同部分的多个摘要，请将它们整合成一个连贯、全面的总摘要。
            确保最终摘要涵盖所有重要信息，并且逻辑连贯、结构清晰。

各部分摘要:
{summaries}

整合后的总摘要:"""
        )
        
        # 初始化摘要链
        self.summary_chain = self.summary_prompt | self.llm
        self.merge_chain = self.merge_prompt | self.llm
    
    def _chunk_text(self, text: str, max_chunk_size: int = 4000) -> List[str]:
        """
        将长文本分割成适合LLM处理的小块
        
        Args:
            text: 需要分割的文本
            max_chunk_size: 每块的最大字符数
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        # 简单按段落分割
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # 如果当前段落加上当前块不超过最大大小，则添加到当前块
            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                # 如果当前块不为空，添加到chunks
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个段落超过最大块大小，需要进一步分割
                if len(para) > max_chunk_size:
                    # 按句子分割
                    sentences = para.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n")
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                            if current_chunk:
                                current_chunk += " "
                            current_chunk += sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            # 如果单个句子超过最大块大小，直接截断
                            if len(sentence) > max_chunk_size:
                                # 按最大块大小分割
                                for i in range(0, len(sentence), max_chunk_size):
                                    chunks.append(sentence[i:i+max_chunk_size])
                                current_chunk = ""
                            else:
                                current_chunk = sentence
                else:
                    current_chunk = para
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    async def summarize_document(self, document: Document) -> str:
        """
        对单个Document对象进行摘要
        
        Args:
            document: 文档对象
            
        Returns:
            str: 文档摘要
        """
        try:
            return await self.summarize_text(document.page_content)
        except Exception as e:
            logger.error(f"摘要生成失败: {str(e)}")
            return "摘要生成失败"
    
    async def summarize_text(self, text: str) -> str:
        """
        对文本内容进行摘要
        
        Args:
            text: 文本内容
            
        Returns:
            str: 文本摘要
        """
        try:
            # 检查文本长度，如果较短则直接摘要
            if len(text) < 4000:
                return await self.summary_chain.ainvoke({"content": text})
            
            # 对长文本进行分块处理
            chunks = self._chunk_text(text)
            logger.info(f"文本已分割为{len(chunks)}个块进行摘要")
            
            # 对每个块生成摘要
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                try:
                    summary = await self.summary_chain.ainvoke({"content": chunk})
                    chunk_summaries.append(summary)
                    logger.info(f"已完成第{i+1}/{len(chunks)}块的摘要生成")
                except Exception as chunk_error:
                    logger.error(f"处理第{i+1}块时出错: {str(chunk_error)}")
                    # 添加一个错误占位符，确保索引一致性
                    chunk_summaries.append(f"[此部分摘要生成失败]")
            
            # 如果只有一个块的摘要，直接返回
            if len(chunk_summaries) == 1:
                return chunk_summaries[0]
            
            # 合并多个块的摘要
            combined_summaries = "\n\n".join([f"部分{i+1}:\n{summary}" for i, summary in enumerate(chunk_summaries)])
            
            # 如果合并后的摘要仍然很长，可能需要再次分块处理
            if len(combined_summaries) > 4000:
                logger.info("合并后的摘要仍然很长，进行二次摘要")
                # 再次分块并摘要
                secondary_chunks = self._chunk_text(combined_summaries)
                secondary_summaries = []
                
                for i, chunk in enumerate(secondary_chunks):
                    try:
                        summary = await self.merge_chain.ainvoke({"summaries": chunk})
                        secondary_summaries.append(summary)
                    except Exception as chunk_error:
                        logger.error(f"二次摘要第{i+1}块时出错: {str(chunk_error)}")
                        secondary_summaries.append(f"[此部分二次摘要生成失败]")
                
                # 最终合并
                final_combined = "\n\n".join(secondary_summaries)
                return await self.merge_chain.ainvoke({"summaries": final_combined})
            else:
                # 直接合并所有摘要
                return await self.merge_chain.ainvoke({"summaries": combined_summaries})
                
        except Exception as e:
            logger.error(f"摘要生成过程中出错: {str(e)}")
            return "摘要生成失败"
    
    async def summarize_documents(self, documents: List[Document]) -> str:
        """
        对多个Document对象进行摘要
        
        Args:
            documents: 文档对象列表
            
        Returns:
            str: 合并后的文档摘要
        """
        try:
            # 合并所有文档内容
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            return await self.summarize_text(combined_text)
        except Exception as e:
            logger.error(f"多文档摘要生成失败: {str(e)}")
            return "多文档摘要生成失败"
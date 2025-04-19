from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class TableToTextConverter:
    """将表格转换为自然语言描述"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["table_content"],
            template="""
            你是一个专业的数据分析师。请将下面的表格数据转换为详细的自然语言描述，
            使其易于理解和检索。描述应包括表格的主要内容、数据之间的关系以及可能的见解。
            
            表格内容:
            {table_content}
            
            自然语言描述:
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def convert(self, table_content):
        """将表格内容转换为自然语言描述"""
        return self.chain.run(table_content=table_content)
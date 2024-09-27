import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def rewrite_query(original_query, llm_chain):
    response = llm_chain.invoke(original_query)
    return response.content

def generate_step_back_query(original_query, llm_chain):
    response = llm_chain.invoke(original_query)
    return response

def decompose_query(original_query, llm_chain):
    response = llm_chain.invoke(original_query).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startwith('Sub-queries:')]
    return sub_queries

class RAGQueryProcessor:
    def __init__(self):
        # 初始化llm
        self.rewrite_llm = ChatOpenAI(temperature=0, model_name='gpt-4o', max_tokens=4000)
        self.step_back_llm = ChatOpenAI(temperature=0, model_name='gpt-4o', max_tokens=4000)
        self.sub_query_llm = ChatOpenAI(temperature=0, model_name='gpt-4o', max_tokens=4000)

        query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

        Original query: {original_query}

        Rewritten query:"""

        step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
        Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

        Original query: {original_query}

        Step-back query:"""

        subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
        Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

        Original query: {original_query}

        example: What are the impacts of climate change on the environment?

        Sub-queries:
        1. What are the impacts of climate change on biodiversity?
        2. How does climate change affect the oceans?
        3. What are the effects of climate change on agriculture?
        4. What are the impacts of climate change on human health?"""

        # 创建LLMChains
        self.query_rewriter = PromptTemplate(input_variables=["original_query"], template=query_rewrite_template) | self.rewrite_llm
        self.step_back_chain = PromptTemplate(input_variables=["original_query"], template=step_back_template) | self.step_back_llm
        self.subquery_decomposer_chain = PromptTemplate(input_variables=["original_query"], template=subquery_decomposition_template) | self.sub_query_llm

    def run(self, original_query):
        print("Original_query:", original_query)
        rewritten_query = rewrite_query(original_query, self.query_rewriter)
        step_back_query = generate_step_back_query(original_query, self.step_back_chain)
        sub_queries = decompose_query(original_query, self.subquery_decomposer_chain)

        print("\nRewritten query:", rewrite_query)
        print("\nStep back query:", step_back_query)
        print("\nSub queries:\n")
        for i, sub_query in enumerate(sub_queries):
            print(f"{i}.{sub_query}")
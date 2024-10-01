import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# 获取api_key
load_dotenv()
os.environ["OPEN_API_KEY"] = os.getenv('OPEN_API_KEY')

class HyDeRetriever:
    def __init__(self, files_path, chunk_size=500, chunk_overlop=100):
        self.llm = ChatOpenAI(temperature=0, model_name="gtp-4o", max_tokens=4000)
        self.embeddings = OpenAIEmbeddings()
        self.chunk_size = chunk_size
        self.chunk_overlop = chunk_overlop
        self.vectorstore = encode_pdf(files_path, self.chunk_size, self.chunk_overlop)

        self.hyde_prompt = PromptTemplate(
            input_variables=['query', 'chunk_size'],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            The document size has to be exactly {chunk_size} characters."""
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    # 生成假设性文档
    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content
    
    # 在faiss中检索文档，返回检索结果
    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k)
        return similar_docs, hypothetical_doc


# pdf编码生成向量索引
def encode_pdf(files_path, chunk_size, chunk_overlop):
    loader = PyPDFLoader(files_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlop=chunk_overlop, length_function=len
    )
    text = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore


# 数据清洗
def replace_t_with_space(list_of_documents):
    for doc in list_of_documents:
        doc.page_content = doc.pagement.replace('\t', ' ')
    return list_of_documents
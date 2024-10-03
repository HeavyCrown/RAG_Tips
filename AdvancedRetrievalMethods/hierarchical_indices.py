import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evalute_rag import *

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

async def encode_pdf_hierarchical(path, chunk_size=1000, chunk_overlop=200, is_string=False):
    # 加载文档
    loader = PyPDFLoader(path)
    documents = await asyncio.to_thread(loader.load)

    # 构建摘要链
    summary_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=4000)
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")

    # 构造异步summarize函数，返回一个Document类型
    async def summarize_doc(doc):
        summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
        summary = summary_output['output_text']
        return Document(page_content=summary, metadata={"source": path, "page": doc.metadata["page"], "summary": True})
    
    # 得到所有document的summary，每个batch启动一个线程，并行的完成所有异步调用
    summarizes = []
    batch_size = 5
    for i in range(0, len(documents), batch_size):
        batch = documents[i: i+batch_size]
        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
        summarizes.extend(batch_summaries)
        await asyncio.sleep(1)
    
    # 划分chunk并更新它的metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlop, len)
    detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)
    for i, chunk in enumerate(detailed_chunks):
        chunk.metadata.update({"chunk_id": i, "summary": False, "page": int(chunk.metadata.get("page", 0))})
    
    # 创建层级index的两个vecstore
    embeddings = OpenAIEmbeddings()

    async def create_vectorstore(docs):
        return await retry_with_exponential_backoff(asyncio.to_thread(FAISS.from_documents, docs, embeddings))
    
    summary_vecstore, detailed_vecstore = await asyncio.gather(
        create_vectorstore(summarizes),
        create_vectorstore(detailed_chunks)
    )

    return summary_vecstore, detailed_vecstore


# 层级检索
def retrieve_hierarchical(query, summary_vecstore, detailed_vecstore, k_summaries=3, k_chunks=5):
    top_summaries = summary_vecstore.similarity_search(query, k_summaries)
    relevant_chunks = []
    for summary in top_summaries:
        page_number = summary.metadata["page"]
        page_filter = lambda metadata: metadata["page"] == page_number
        page_chunks = detailed_vecstore.similarity_search(query, k_chunks, page_filter)
        relevant_chunks.extend(page_chunks)
    return relevant_chunks

class HierarchicalRAG:
    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.summary_vecstore = None
        self.detailed_vecstore = None

    async def run(self, query):
        if os.path.exists("../vector_stores/summary_store") and os.path.exists("../vector_stores/detailed_store"):
            embeddings = OpenAIEmbeddings()
            self.summary_store = FAISS.load_local("../vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
            self.detailed_store = FAISS.load_local("../vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)
        else:
            self.summary_store, self.detailed_store = await encode_pdf_hierarchical(self.pdf_path, self.chunk_size, self.chunk_overlap)
            self.summary_store.save_local("../vector_stores/summary_store")
            self.detailed_store.save_local("../vector_stores/detailed_store")

        results = retrieve_hierarchical(query, self.summary_store, self.detailed_store)
        for chunk in results:
            print(f"Page: {chunk.metadata['page']}")
            print(f"Content: {chunk.page_content}...")
            print("---")
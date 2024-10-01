import os
import sys
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evalute_rag import *


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def encode_pdf_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts

def create_bm25_index(documents:List[Document]) -> BM25Okapi:
    """
    从documents中创建BM25的索引
    """
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    结合基于关键词的 BM25 搜索和基于向量的搜索进行融合检索

    Args:
        vectorstore: 一个向量存储库，包含文档。
        bm25: 预先计算好的 BM25 索引。
        query: 查询字符串。
        k: 要检索的文档数量。
        alpha: 向量搜索得分的权重（1-alpha 将是 BM25 得分的权重）。

    Returns:
        基于 combined_scores 排名的前 k 个文档列表。
    """
    # 首先得到vectorestore的所有文档
    all_docs = vectorstore.similarity_search(" ", k=vectorstore.index.ntotal)
    
    # 执行BM25和向量搜索并得到得分
    bm25_scores = bm25.get_scores(query.split())
    vectore_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    # 将得分归一化
    vector_scores = np.array([score for _, score in vectore_results])
    vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    # 通过alpha作为权重计算两种检索的融合得分并由大到小排序
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    return [all_docs[i] for i in sorted_indices[:k]]

# 使用实例
if __name__ == "__main__":
    path = ""
    vectorstore, cleaned_texts = encode_pdf_and_get_split_documents(path)
    bm25 = create_bm25_index(cleaned_texts)
    query = ""

    # 进行融合检索
    top_docs = fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5)
    docs_content = [doc.page_content for doc in top_docs]
    show_context(docs_content)
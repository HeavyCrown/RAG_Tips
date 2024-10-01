import random
import time
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.evaluation import DatasetGenerator, FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class RAGEvaluator:
    def __init__(self, data_dir, num_eval_questions, chunk_sizes):
        self.data_dir = data_dir
        self.num_eval_questions = num_eval_questions
        self.chunk_sizes = chunk_sizes
        self.documents = self.load_documents()
        self.eval_questions = self.generate_eval_questions()
        self.service_context = self.create_service_context()
        self.faithfulness_evaluator = self.create_faithfulness_evaluator()
        self.relevancy_evaluator = self.create_relevancy_evaluator()

    def load_documents(self):
        return SimpleDirectoryReader(self.data_dir).load_data()

    def generate_eval_questions(self):
        eval_documents = self.documents[0:20]
        data_generator = DatasetGenerator.from_documents(eval_documents)
        eval_questions = data_generator.generate_questions_from_nodes()
        return random.sample(eval_questions, self.num_eval_questions)

    def create_service_context(self):
        llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")
        return ServiceContext.from_defaults(llm=llm)

    def create_faithfulness_evaluator(self):
        faithfulness_evaluator = FaithfulnessEvaluator(service_context=self.service_context)
        faithfulness_template = PromptTemplate(
            """
            请说明某条信息是否直接得到上下文的支持。
            你需要回答“是”或“不是”。
            如果上下文的任何部分明确支持该信息，即使大部分上下文是不相关的，也要回答"是"。如果上下文没有明确支持该信息，则回答“否”。
            """
        )
        faithfulness_evaluator.update_prompts({"your_prompt_key": faithfulness_template})
        return faithfulness_evaluator

    def create_relevancy_evaluator(self):
        relevancy_evaluator = RelevancyEvaluator(service_context=self.service_context)
        return relevancy_evaluator

    def evaluate(self, chunk_size):
        total_response_time = 0
        total_faithfulness = 0
        total_relevancy = 0

        # 创建向量索引
        llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
        service_context = ServiceContext.from_defaults(llm=llm, chunk_size=chunk_size,
                                                       chunk_overlop=chunk_size // 5)
        vector_index = VectorStoreIndex.from_documents(self.documents[0:20], service_context=service_context)

        # 创建查询
        query_engine = vector_index.as_query_engine(similarity_top_k=5)
        num_questions = len(self.eval_questions)

        # 迭代每一个问题做评估
        for question in self.eval_questions:
            start_time = time.time()
            response_vector = query_engine.query(question)
            elapsed_time = time.time() - start_time

            faithfulness_result = self.faithfulness_evaluator.evaluate_response(response=response_vector)
            relevancy_result = self.relevancy_evaluator.evaluate_response(response=response_vector)

            total_response_time += elapsed_time
            total_faithfulness += faithfulness_result
            total_relevancy += relevancy_result

        # 计算平均响应时间，平均忠诚度和平均相关度三项指标
        average_response_time = total_response_time / num_questions
        average_faithfulness = total_faithfulness / num_questions
        average_relevancy = total_relevancy / num_questions

        return average_response_time, average_faithfulness, average_relevancy


if __name__ == "__main__":
    data_dir = ''
    chunk_sizes = [128, 256]
    num_questions = 10
    evaluator = RAGEvaluator(data_dir, num_questions, chunk_sizes)
    for chunk_size in chunk_sizes:
        evaluator.evaluate(chunk_size)

import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


ELASTIC_URL = os.getenv("ELASTIC_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME=os.getenv('INDEX_NAME')

es_client = Elasticsearch(ELASTIC_URL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def elastic_search_hybrid(field, query, vector, company_id, index_name=INDEX_NAME):
    knn_query = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "boost": 0.5,
        "filter": {
            "term": {
                "company_id": company_id
            }
        }
    }

    keyword_query = {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question", "context"],
                    "type": "best_fields",
                    "boost": 0.5,
                }
            },
            "filter": {
                "term": {
                    "company_id": company_id
                }
            }
        }
    }

    search_query = {
        "knn": knn_query,
        "query": keyword_query,
        "size": 5,
        "_source": ["question", "context", "company_id", "doc_id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs


def build_prompt(query, search_results):
    prompt_template = """
    You're a course financial assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT: 
    {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"question: {doc['question']}\nanswer: {doc['context']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt, model='gpt-4o-mini'):
    start_time = time.time()

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    tokens = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }

    end_time = time.time()
    response_time = end_time - start_time
    
    return answer, tokens, response_time


def evaluate_relevance(question, answer):
    llm_as_judge_prompt = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer compared to the given question provided.
    Based on the relevance and similarity of the generated answer to the given question, you will classify
    it as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the original
    answer and provide your evaluation in parsable JSON without using code blocks:

    {{
    "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
    "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = llm_as_judge_prompt.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt, 'gpt-4o-mini')
    
    try:
        json_eval = json.loads(evaluation)
        return json_eval['Relevance'], json_eval['Explanation'], tokens
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", tokens


def calculate_openai_cost(tokens):
    openai_cost = 0

    openai_cost = (tokens['prompt_tokens'] * 0.03 + tokens['completion_tokens'] * 0.06) / 1000

    return openai_cost


def get_answer(query, company_id):

    vector = model.encode(query)
    search_results = elastic_search_hybrid('question_context_vector', query, vector, company_id)

    prompt = build_prompt(query, search_results)
    answer, tokens, response_time = llm(prompt)
    
    relevance, explanation, eval_tokens = evaluate_relevance(query, answer)

    openai_cost = calculate_openai_cost(tokens)
 
    return {
        'answer': answer,
        'response_time': response_time,
        'relevance': relevance,
        'relevance_explanation': explanation,
        'prompt_tokens': tokens['prompt_tokens'],
        'completion_tokens': tokens['completion_tokens'],
        'total_tokens': tokens['total_tokens'],
        'eval_prompt_tokens': eval_tokens['prompt_tokens'],
        'eval_completion_tokens': eval_tokens['completion_tokens'],
        'eval_total_tokens': eval_tokens['total_tokens'],
        'openai_cost': openai_cost
    }
# financial-faq-assistant-llm-rag-vector-search

```bash
│ 
├── .env
├── dashboard_ss.png
├── Financial FAQ Bot Dashboard.json
│ 
├── ground-truth-data.csv
├── documents-with-ids.json
├── documents-with-vectors.json
│ 
├── evaluate-rag.ipynb
├── evaluate-retrieval.ipynb
├── generate_ground_truth.ipynb
│ 
│ 
├── app
│   ├── .env
│   ├── app.py
│   ├── assistant.py
│   ├── db.py
│   ├── prep.py
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   ├── requirements.txt
│
├── data
│   ├── results-gpt4o-evaluations.csv
│   ├── results-gpt4o.csv
│   ├── results-gpt35-evaluations.csv
│   ├── results-gpt35.csv
```

## Project Description

The goal of this project is to improve the accuracy and relevance of financial information retrieval related to high-margin, high-value companies from the stock exchange. Initially, when using large language models (LLMs) to access such financial data, significant issues were encountered. The models often hallucinated, providing either inaccurate information or details that were irrelevant to the specific query.

To address this, I utilized the Financial QA dataset, which contains over 10,000 questions and answers related to financial filings and reports of 69 different companies. My objective was to retrieve precise and relevant information about these companies, leveraging their filings and reports to generate company-specific insights.

This project aims to retrieve relevant information from this dataset using Retrieval-Augmented Generation (RAG), a method that combines retrieval techniques with generative models. By implementing RAG, the goal was to ensure that the generated responses were not only accurate but also directly related to the queried company.

### Project Steps:
#### Generating Ground Truth Data: 
Preparing a dataset with ground-truth information to evaluate the accuracy of the retrieved responses.

#### Evaluating Retrieval Methods: 
Testing and comparing various retrieval strategies to assess their performance in providing relevant financial data.

#### Assessing RAG Performance: 
Evaluating the RAG method with different models to measure how well it retrieves and generates accurate, company-specific information.
Developing an Interface: Creating a user-friendly interface to facilitate seamless querying and retrieval of financial data.

By the end of this project, the aim is to build a robust retrieval system capable of extracting specific, accurate financial details from company filings, effectively mitigating the issues of hallucination and irrelevant information that are common in LLMs.

### Generation of Ground Truth Data
To generate the ground truth data, I designed a prompt template specifically for ChatGPT. The goal was to create multiple variations of financial questions related to different companies. The prompt instructs ChatGPT to emulate a user asking financial questions and, based on the original question and a specified company financial quote, generate three similar but distinct questions about the same topic. Importantly, the generated questions must contain the original company name rather than the financial quote.

The prompt template I used is as follows:

```json
You emulate someone who is asking financial questions. 
Based on the following original question and the specified company financial quote, generate 3 different but similar questions that a user might ask about the same topic.
Questions generated should contain company original name not quote.

ORIGINAL QUESTION: {question} 
COMPANY QUOTE: {company_id} 

Provide the output in parsable JSON format without using code blocks. Below is the right output format:

["question1", "question2", "question3"]
```

### Evaluating Different Retrieval Methods
In the second step, I used the ground truth data along with the original dataset to generate vector representations for each data point. For every entry, I created vector embeddings for the  `question`, `context`, and the `question-context` combination. I used the `sentence-transformers` library's pre-trained model, `multi-qa-MiniLM-L6-cos-v1`, to generate these embeddings. This model was applied to vectorize both the original questions and answers, as well as each entry from the ground truth data.

Once the original data was embedded, I stored these vectors in `Elasticsearch` and retrieved the five closest matching entries for each ground-truth entry. To assess the accuracy of the retrieval, I checked whether the original entry of the ground truth data was present among the top 5 retrieved results. I then evaluated performance based on `hit rate` and `Mean Reciprocal Rank (MRR)`.

After testing nearly ten different retrieval methods, the best-performing approach was selected for the final application. The chosen method is `hybrid search, which combines vector-based and keyword-based search`. Specifically, it uses `question-context vector embeddings` to achieve the most accurate and relevant retrievals.


### Assessing Performances of Different RAG Models
In the third step, having established that relevant results were being retrieved and with the best retrieval method determined, the focus shifted to evaluating the performance of `large language models (LLMs)` in answering questions using the retrieved data. Using the ground truth data, I assessed two different OpenAI models: `GPT-3.5` and `GPT-4o-mini`. For each ground truth entry, and the top 5 similar entries retrieved from Elasticsearch, the models were tasked with generating answers. The generated answers were then evaluated against the original answers from the dataset.

To assess the similarity between the generated and original answers, I calculated the `cosine similarity` between the embeddings of both answers. Additionally, I employed `GPT-4o-mini` as a `"LLM as Judge"` to classify the answers generated by both models as either relevant, partly relevant, or irrelevant. This allowed for a human-like evaluation of the generated responses' quality. The final results showed that the `GPT-4o-mini` model performed slightly better than GPT-3.5 in terms of answer quality, with its generated answers being closer to the original answers, as measured by both cosine similarity and the LLM-as-judge classification results.

```json
count    2100.000000
mean        0.630666
std         0.252160
min        -0.074854
25%         0.432983
50%         0.668283
75%         0.831950
max         1.000000
Name: cosine, dtype: float64

Relevance
RELEVANT           172
PARTLY_RELEVANT     26
NON_RELEVANT        12
Name: count, dtype: int64
```

By utilizing both quantitative measures (cosine similarity) and qualitative judgment (LLM-as-judge), the GPT-4o-mini model was found to provide higher-quality responses in this financial question-answering task.
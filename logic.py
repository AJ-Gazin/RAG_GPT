import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from firecrawl import FirecrawlApp
from tiktoken import encoding_for_model
from prompts import answer_query_prompt, score_hyperlinks_prompt
import re
import faiss
import numpy as np

# Global variables to cache API keys and FAISS index
firecrawl_api_key = None
openai_client = None
faiss_index = None
document_store = {}

def set_api_keys(firecrawl_key, openai_key):
    global firecrawl_api_key, openai_client, faiss_index
    firecrawl_api_key = firecrawl_key
    openai_client = OpenAI(api_key=openai_key)
    faiss_index = faiss.IndexFlatL2(1536)  # Assuming embedding size is 1536
    return "API keys have been set."

def generate_embeddings(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

def truncate_text_for_embedding(text, max_tokens=8192, model="text-embedding-ada-002"):
    enc = encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

def store_embeddings(document_path):
    """
    Generates and stores embeddings for the given document.
    """
    global faiss_index, document_store
    
    with open(document_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    truncated_text = truncate_text_for_embedding(document_text)
    embedding = generate_embeddings(truncated_text)
    faiss_index.add(np.array([embedding]))
    document_store[faiss_index.ntotal - 1] = truncated_text

def parse_webpage(url, firecrawl_api_key):
    folder_name = 'parsed_pages'
    original_file_path = os.path.join(folder_name, f"{url.replace('http://', '').replace('https://', '').replace('/', '_')}.txt")
    hyperlinks_file_path = os.path.join(folder_name, 'hyperlinks.txt')

    # Check if the webpage has already been parsed
    if os.path.exists(original_file_path):
        hyperlinks_with_context = read_existing_hyperlinks(hyperlinks_file_path)
        return original_file_path, hyperlinks_with_context
    
    # Firecrawl to fetch and process the webpage
    crawler = FirecrawlApp(api_key=firecrawl_api_key)
    try:
        crawl_result = crawler.crawl_url(url, {'crawlerOptions': {'excludes': ['blog/*']}})
    except Exception as e:
        print(f"Firecrawl job failed: {e}")
        return None, None
    
    if not crawl_result:
        raise ValueError(f"Unexpected or empty response format: {crawl_result}")
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    with open(original_file_path, 'w', encoding='utf-8') as f:
        for result in crawl_result:
            f.write(result.get('markdown', '') + "\n\n")
    
    # Use BeautifulSoup to extract hyperlinks and context
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    
    page_content = response.text
    soup = BeautifulSoup(page_content, 'html.parser')
    new_hyperlinks_with_context = extract_hyperlinks_with_context(soup)
    
    # Read existing hyperlinks and contexts
    existing_hyperlinks = read_existing_hyperlinks(hyperlinks_file_path)
    
    # Update existing hyperlinks with new contexts
    for link, context in new_hyperlinks_with_context:
        if link in existing_hyperlinks:
            if context not in existing_hyperlinks[link]:
                existing_hyperlinks[link].append(context)
        else:
            existing_hyperlinks[link] = [context]
    
    # Write updated hyperlinks with contexts
    with open(hyperlinks_file_path, 'w', encoding='utf-8') as f:
        for link, contexts in existing_hyperlinks.items():
            for context in contexts:
                f.write(f"{link}\ncontext: {context}\n")
    
    return original_file_path, new_hyperlinks_with_context

def read_existing_hyperlinks(file_path):
    """
    Reads existing hyperlinks and their contexts from the given file path.
    Returns a dictionary with hyperlinks as keys and lists of contexts as values.
    """
    hyperlinks = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            current_link = None
            for line in lines:
                if line.startswith('http'):
                    current_link = line
                    if current_link not in hyperlinks:
                        hyperlinks[current_link] = []
                elif line.startswith('context:') and current_link:
                    context = line.replace('context:', '').strip()
                    hyperlinks[current_link].append(context)
    return hyperlinks

def extract_hyperlinks_with_context(soup):
    """
    Extracts hyperlinks and their surrounding context (sentence) from the BeautifulSoup object.
    Returns a list of tuples containing (hyperlink, context).
    """
    hyperlinks_with_context = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Find the surrounding sentence
        sentence = get_surrounding_sentence(a_tag)
        if sentence:
            hyperlinks_with_context.append((href, sentence))
    return hyperlinks_with_context

def get_surrounding_sentence(a_tag):
    """
    Finds and returns the sentence containing the given <a> tag.
    """
    text = a_tag.find_parent().get_text(separator=' ')
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    for sentence in sentences:
        if a_tag.get_text() in sentence:
            return sentence.strip()
    return None

def setup_rag(parsed_files):
    """
    Reads the content of the given files and returns a list of documents.
    """
    documents = [open(file, 'r', encoding='utf-8').read() for file in parsed_files]
    return documents

def truncate_documents(documents, max_tokens, model="gpt-3.5-turbo"):
    """
    Truncates the documents to fit within the maximum token limit of the model.
    """
    enc = encoding_for_model(model)
    token_count = 0
    truncated_docs = []
    
    for doc in documents:
        doc_tokens = enc.encode(doc)
        if token_count + len(doc_tokens) > max_tokens:
            truncated_docs.append(enc.decode(doc_tokens[:max_tokens - token_count]))
            break
        truncated_docs.append(doc)
        token_count += len(doc_tokens)
    
    return truncated_docs

def answer_query(query, primary_document, relevant_document):
    """
    Answers the given query using the provided documents and logs the prompt for debugging.
    """
    combined_document = primary_document + "\n\n" + relevant_document
    max_context_length = 16385  # Adjust based on the model's max context length
    truncated_documents = truncate_documents([combined_document], max_context_length)
    truncated_combined_document = "\n\n".join(truncated_documents)
    
    prompt = answer_query_prompt(query, truncated_combined_document)
    
    # Log the prompt for debugging
    with open('prompt_log.txt', 'a', encoding='utf-8') as log_file:
        log_file.write(f"Prompt sent to OpenAI:\n{prompt}\n{'-'*80}\n")
    
    response = openai_client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo"
    )
    answer = response.choices[0].message.content
    return answer

def score_hyperlinks(query, hyperlinks_with_context):
    """
    Scores the relevancy of each hyperlink based on its context and the user's query.
    """
    prompt = score_hyperlinks_prompt(query, hyperlinks_with_context)
    response = openai_client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo"
    )
    scores = response.choices[0].message.content
    return parse_scores(scores)

def parse_scores(scores):
    """
    Parses the scores returned by the LLM.
    """
    score_dict = {}
    lines = scores.split('\n')
    for line in lines:
        if line.startswith('http'):
            parts = line.split(' ')
            url = parts[0]
            score = float(parts[-1])
            score_dict[url] = score
    return score_dict

def process_inputs(url, query):
    global firecrawl_api_key
    
    if not firecrawl_api_key or not openai_client:
        return "Please set the API keys first."
    
    # Parse the original webpage and get hyperlinks with context
    original_file_path, hyperlinks_with_context = parse_webpage(url, firecrawl_api_key)
    
    if not original_file_path or not hyperlinks_with_context:
        return "Failed to parse the original webpage."
    
    # Store embeddings for the primary document
    store_embeddings(original_file_path)
    
    # Read the primary document
    primary_document = open(original_file_path, 'r', encoding='utf-8').read()
    
    # Score the relevancy of each hyperlink
    scores = score_hyperlinks(query, hyperlinks_with_context)
    
    # Find the most relevant hyperlink
    most_relevant_url = max(scores, key=scores.get)
    high_relevancy_threshold = 0.9  # Define a high relevancy threshold
    
    relevant_document = ""
    if scores[most_relevant_url] > high_relevancy_threshold:
        # Parse the most relevant hyperlink
        relevant_file_path, _ = parse_webpage(most_relevant_url, firecrawl_api_key)
        
        if relevant_file_path:
            # Store embeddings for the relevant document
            store_embeddings(relevant_file_path)
            relevant_document = open(relevant_file_path, 'r', encoding='utf-8').read()
    
    # Answer the query using the primary and relevant documents
    answer = answer_query(query, primary_document, relevant_document)
    
    return answer

import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from firecrawl import FirecrawlApp
from tiktoken import encoding_for_model
from prompts import answer_query_prompt, score_hyperlinks_prompt
import re

# Global variables to cache API keys
firecrawl_api_key = None
openai_client = None

def set_api_keys(firecrawl_key, openai_key):
    global firecrawl_api_key, openai_client
    firecrawl_api_key = firecrawl_key
    openai_client = OpenAI(api_key=openai_key)
    return "API keys have been set."

def parse_webpage(url, firecrawl_api_key):
    folder_name = 'parsed_pages'
    original_file_path = os.path.join(folder_name, f"{url.replace('http://', '').replace('https://', '').replace('/', '_')}.txt")
    hyperlinks_file_path = os.path.join(folder_name, 'hyperlinks.txt')

    # Check if the webpage has already been parsed
    if os.path.exists(original_file_path):
        print("Webpage has already been parsed. Using existing data.")
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
    
    print("Extracted hyperlinks with context:", new_hyperlinks_with_context)  # Debug statement
    
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
    
    # Debug statement to confirm hyperlinks have been written
    with open(hyperlinks_file_path, 'r', encoding='utf-8') as f:
        saved_hyperlinks = f.read().splitlines()
        print("Saved hyperlinks:", saved_hyperlinks)
    
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

def answer_query(query, documents):
    """
    Answers the given query using the provided documents.
    """
    combined_document = "\n\n".join(documents)
    max_context_length = 4096  # Adjust based on the model's max context length
    truncated_documents = truncate_documents([combined_document], max_context_length)
    truncated_combined_document = "\n\n".join(truncated_documents)
    
    prompt = answer_query_prompt(query, truncated_combined_document)
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
    
    documents = setup_rag([original_file_path])
    
    # Answer the query using the original documents
    answer = answer_query(query, documents)
    
    # Score the relevancy of each hyperlink
    scores = score_hyperlinks(query, hyperlinks_with_context)
    
    # Find the most relevant hyperlink
    most_relevant_url = max(scores, key=scores.get)
    high_relevancy_threshold = 0.9  # Define a high relevancy threshold
    
    if scores[most_relevant_url] > high_relevancy_threshold:
        print(f"Most relevant URL: {most_relevant_url} with score {scores[most_relevant_url]}")
        # Parse the most relevant hyperlink and add it to RAG
        relevant_file_path, _ = parse_webpage(most_relevant_url, firecrawl_api_key)
        
        if relevant_file_path:
            documents += setup_rag([relevant_file_path])
            answer = answer_query(query, documents)
    
    return answer

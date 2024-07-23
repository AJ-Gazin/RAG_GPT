import os
import openai
from firecrawl import FirecrawlApp
from tiktoken import encoding_for_model

def parse_webpage(url, firecrawl_api_key):
    folder_name = 'parsed_pages'
    original_file_path = os.path.join(folder_name, f"{url.replace('http://', '').replace('https://', '').replace('/', '_')}.txt")
    
    if os.path.exists(original_file_path):
        with open(original_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            hyperlinks = extract_hyperlinks(content)
        return original_file_path, hyperlinks
    
    crawler = FirecrawlApp(api_key=firecrawl_api_key)
    crawl_result = crawler.crawl_url(url, {'crawlerOptions': {'excludes': ['blog/*']}})
    
    if not isinstance(crawl_result, list) or len(crawl_result) == 0:
        raise ValueError(f"Unexpected or empty response format: {crawl_result}")
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    with open(original_file_path, 'w', encoding='utf-8') as f:
        for result in crawl_result:
            f.write(result.get('markdown', '') + "\n\n")
    
    hyperlinks = [link['url'] for link in crawl_result[0].get('hyperlinks', [])]
    
    return original_file_path, hyperlinks

def extract_hyperlinks(content):
    # Implement a simple way to extract hyperlinks from the saved content
    return []

def get_relevant_hyperlinks(openai_api_key, query, hyperlinks):
    client = openai.OpenAI(api_key=openai_api_key)
    
    prompt = f"User query: {query}\n\nHyperlinks:\n" + "\n".join(hyperlinks) + "\n\nBased on the user query, which of these hyperlinks are relevant? List the relevant hyperlinks."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    relevant_hyperlinks = response.choices[0].message.content.strip().split("\n")
    
    return [link.strip() for link in relevant_hyperlinks if link.strip() in hyperlinks]

def follow_and_parse_hyperlinks(relevant_hyperlinks, firecrawl_api_key):
    documents = []
    crawler = FirecrawlApp(api_key=firecrawl_api_key)
    
    for link in relevant_hyperlinks:
        try:
            link_data = crawler.scrape_url(link)
            if 'data' not in link_data or 'content' not in link_data['data']:
                print(f"Failed to scrape {link}: {link_data}")
                continue
            link_text = link_data['data']['content']
            documents.append(link_text)
        except Exception as e:
            print(f"Failed to parse {link}: {e}")
    
    return documents

def setup_rag(openai_api_key, parsed_files):
    openai.api_key = openai_api_key
    
    documents = []
    for file in parsed_files:
        with open(file, 'r', encoding='utf-8') as f:
            documents.append(f.read())

    return documents

def truncate_documents(documents, max_tokens, model="gpt-3.5-turbo"):
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

def answer_query(query, documents, openai_api_key):
    combined_document = "\n\n".join(documents)
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Truncate the combined document to fit within the model's context length
    max_context_length = 4096  # Adjust based on the model's max context length
    truncated_documents = truncate_documents([combined_document], max_context_length)
    truncated_combined_document = "\n\n".join(truncated_documents)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on the following documents, please answer: {query}"},
            {"role": "assistant", "content": truncated_combined_document}
        ]
    )
    answer = response.choices[0].message.content
    return answer

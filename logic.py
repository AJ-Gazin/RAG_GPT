import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from openai import OpenAI
import tiktoken
from datetime import datetime
import logging
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import networkx as nx
from pyvis.network import Network

from dotenv import load_dotenv
from prompts import answer_prompt

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MAX_TOKENS = 128000
MAX_OUTPUT_TOKENS = 16000
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tokenizer = tiktoken.get_encoding("cl100k_base")
summary_store = {}

# Configure global settings
Settings.llm = LlamaOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
Settings.chunk_size = 1024
Settings.chunk_overlap = 20

def count_tokens(text):
    return len(tokenizer.encode(text))

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def prioritize_pages(links):
    priority_keywords = ['about', 'services', 'products', 'contact', 'team', 'history']
    return sorted(links, key=lambda link: any(keyword in link.lower() for keyword in priority_keywords), reverse=True)

def crawl_page(current_url, visited, to_visit, max_tokens, total_tokens):
    try:
        response = requests.get(current_url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
            text = f"URL: {current_url}\n\n{extract_text_from_html(response.text)}"
            text_tokens = count_tokens(text)
            if total_tokens + text_tokens > max_tokens:
                logging.warning("Token limit reached, stopping crawl.")
                return None, total_tokens
            summary_store[current_url] = text
            total_tokens += text_tokens
            soup = BeautifulSoup(response.text, 'html.parser')
            new_links = [
                urljoin(current_url, link['href'])
                for link in soup.find_all('a', href=True)
                if urlparse(urljoin(current_url, link['href'])).netloc == urlparse(current_url).netloc
                and urljoin(current_url, link['href']) not in visited
                and urljoin(current_url, link['href']) not in to_visit
            ]
            return prioritize_pages(new_links), total_tokens
    except Exception as e:
        logging.error(f"Error crawling {current_url}: {str(e)}")
    return [], total_tokens

def crawl_website(url, max_pages):
    global summary_store
    summary_store.clear()
    
    logging.info(f"Starting crawl for: {url}")
    visited, to_visit = set(), [url]
    pages_crawled, total_tokens = 0, 0

    while to_visit and pages_crawled < max_pages and total_tokens < MAX_TOKENS:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        new_links, total_tokens = crawl_page(current_url, visited, to_visit, MAX_TOKENS, total_tokens)
        to_visit.extend(new_links or [])
        visited.add(current_url)
        pages_crawled += 1
        logging.info(f"Pages crawled: {pages_crawled}")
        yield pages_crawled, max_pages, f"Crawling: {current_url}"

    yield pages_crawled, max_pages, f"Crawling complete. Pages crawled: {pages_crawled}"

def create_vector_index():
    documents = [Document(text=content) for content in summary_store.values()]
    index = VectorStoreIndex.from_documents(documents)
    
    output_dir = os.getenv("EMBEDDING_DIR", "embeddings")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    embedding_path = f"{output_dir}/{timestamp}"
    index.storage_context.persist(persist_dir=embedding_path)
    
    return index



def generate_graph_visualization(kg_index):
    g = kg_index.get_networkx_graph()

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#222222",
        font_color="white"
    )

    net.from_nx(g)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)
    
    html = net.generate_html()
    html = html.replace("'", "\"")
    
    iframe_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
     display-capture; encrypted-media;" sandbox="allow-modals allow-forms
     allow-scripts allow-same-origin allow-popups
     allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
     allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

    logging.info("Graph visualization HTML generated.")
    return iframe_html

def create_knowledge_graph():
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    documents = [Document(text=content) for content in summary_store.values()]
    
    kg_index = KnowledgeGraphIndex.from_documents(
        documents=documents,
        max_triplets_per_chunk=10,
        storage_context=storage_context,
        include_embeddings=True,
        kg_triplet_extract_fn=kg_triplet_extract_fn
    )
    
    output_dir = os.getenv("GRAPH_DIR", "graphs")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    kg_path = f"{output_dir}/{timestamp}"
    kg_index.storage_context.persist(persist_dir=kg_path)
    
    # Generate and save the graph visualization
    generate_graph_visualization(kg_index)
    
    return kg_index

def kg_triplet_extract_fn(text):
    prompt = f"""
    Extract key information from the following webpage content as a list of triplets in the format (entity1, relation, entity2).
    Focus on main topics, key facts, and relationships between concepts.
    Webpage content:
    {text}
    """
    
    response = client.chat.completions.create(  # Note the correct API method here
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts key information as triplets."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_OUTPUT_TOKENS
    )
    
    triplets = []
    response_text = response.choices[0].message.content
    for line in response_text.split('\n'):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            try:
                triplet = eval(line)
                triplets.append(triplet)
            except Exception as e:
                logging.error(f"Failed to parse line: {line} - Error: {str(e)}")
        else:
            # Attempt to reformat potential triplets
            try:
                triplet = tuple(line.strip('()').split(', '))
                if len(triplet) == 3:
                    triplets.append(triplet)
            except Exception as e:
                logging.error(f"Failed to reformat line: {line} - Error: {str(e)}")
    
    return triplets

def get_latest_dir(parent_dir):
    dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def analyze_website():
    logging.info("Starting analysis process.")
    vector_index = create_vector_index()
    kg_index = create_knowledge_graph()
    graph_html = generate_graph_visualization(kg_index)
    logging.info("Analysis complete.")
    return "Analysis complete.", graph_html


def query_content(query):
    # Load the latest vector index
    vector_dir = os.getenv("EMBEDDING_DIR", "embeddings")
    latest_vector_dir = get_latest_dir(vector_dir)
    vector_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=latest_vector_dir))

    # Load the latest knowledge graph
    kg_dir = os.getenv("GRAPH_DIR", "graphs")
    latest_kg_dir = get_latest_dir(kg_dir)
    kg_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=latest_kg_dir))

    # Create query engines for both RAG and Graph-RAG
    rag_engine = vector_index.as_query_engine()
    graph_rag_engine = kg_index.as_query_engine(include_text=True)

    # Query both engines
    rag_response = rag_engine.query(query)
    graph_rag_response = graph_rag_engine.query(query)

    # Combine responses and prepare context
    combined_context = f"RAG Response: {rag_response}\n\nGraph-RAG Response: {graph_rag_response}"

    # Use the answer_prompt to generate a comprehensive response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes business information."},
            {"role": "user", "content": answer_prompt.format(query=query, context=combined_context)}
        ],
        max_tokens=MAX_OUTPUT_TOKENS
    )

    answer = response.choices[0].message.content

    # Extract URLs from the summary_store for citation
    urls = list(summary_store.keys())

    return answer, ", ".join(urls)

if __name__ == "__main__":
    # You can add any testing or standalone functionality here
    pass

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from openai import OpenAI
import tiktoken
from readability import Document
import logging
import networkx as nx
import plotly.express as px
import spacy
import chromadb
from prompts import summarize_prompt, answer_prompt, extract_entities_prompt
import pandas as pd
import json
from datetime import datetime
from pydantic import BaseModel
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MAX_PAGES = 60
MAX_TOKENS = 128000
EMBEDDING_MAX_TOKENS = 8192
client = None
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize Chroma client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="website_content")

# Initialize spaCy
nlp = spacy.load("en_core_web_trf")

# Initialize NetworkX graph
G = nx.Graph()

# Pydantic models
class Entity(BaseModel):
    name: str
    type: str

class Relationship(BaseModel):
    source: str
    target: str
    type: str

class EntityRelationshipExtraction(BaseModel):
    entities: List[Entity] = []
    relationships: List[Relationship] = []

def count_tokens(text):
    return len(tokenizer.encode(text))

def log_token_usage(input_tokens, output_tokens, api_type):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "api_type": api_type,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
    with open("token_usage_log.jsonl", "a") as log_file:
        json.dump(log_entry, log_file)
        log_file.write("\n")

def extract_relevant_text(html_content):
    doc = Document(html_content)
    return doc.summary()

def get_embeddings(texts, model="text-embedding-3-small"):
    if client is None:
        logging.error("OpenAI client is not initialized. Please set the API key first.")
        return [None] * len(texts)

    embeddings = []
    for text in texts:
        try:
            response = client.embeddings.create(input=[text[:EMBEDDING_MAX_TOKENS]], model=model)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            logging.error(f"Error getting embeddings: {str(e)}")
            embeddings.append(None)
    return embeddings

def extract_entities_and_relationships(text, url):
    if client is None:
        logging.error("OpenAI client is not initialized. Please set the API key first.")
        return EntityRelationshipExtraction()

    try:
        input_text = f"Text: {text}\nURL: {url}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": extract_entities_prompt},
                {"role": "user", "content": input_text}
            ],
            response_format={"type": "json_object"}
        )
        
        if response.choices[0].finish_reason == "stop":
            result = json.loads(response.choices[0].message.content)
            return EntityRelationshipExtraction(**result)
        else:
            logging.error(f"Unexpected finish reason: {response.choices[0].finish_reason}")
            return EntityRelationshipExtraction()
        
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response: {str(e)}")
    except Exception as e:
        logging.error(f"Error extracting entities and relationships: {str(e)}")
    
    return EntityRelationshipExtraction()

def update_knowledge_graph(entities_and_relationships):
    for entity in entities_and_relationships.entities:
        G.add_node(entity.name, type=entity.type)
    
    for relation in entities_and_relationships.relationships:
        G.add_edge(relation.source, relation.target, type=relation.type)

def create_plotly_graph():
    filtered_graph = G.subgraph([node for node in G if G.degree(node) >= 2])
    pos = nx.spring_layout(filtered_graph)
    
    edge_x, edge_y = [], []
    for edge in filtered_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_df = pd.DataFrame({
        'x': [pos[node][0] for node in filtered_graph.nodes()],
        'y': [pos[node][1] for node in filtered_graph.nodes()],
        'text': [f'{node}<br># of connections: {filtered_graph.degree(node)}' for node in filtered_graph.nodes()],
        'size': [5 + filtered_graph.degree(node) for node in filtered_graph.nodes()]
    })
    
    fig = px.scatter(node_df, x='x', y='y', size='size', text='text',
                     title='Website Knowledge Graph',
                     labels={'x': '', 'y': ''},
                     color='size',
                     color_continuous_scale='YlGnBu')
    
    fig.add_trace(px.line(x=edge_x, y=edge_y).data[0])
    
    fig.update_traces(textposition='top center', marker=dict(sizemin=5))
    fig.update_layout(showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    return fig

def crawl_website(url, max_pages, progress=None):
    global G
    G = nx.Graph()
    
    visited = set()
    to_visit = [url]
    pages_crawled = 0
    summary_store = {}

    while to_visit and pages_crawled < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        if progress:
            progress((pages_crawled / max_pages), f"Crawling: {current_url}")
        logging.info(f"Crawling: {current_url}")

        try:
            response = requests.get(current_url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                soup = BeautifulSoup(response.text, 'lxml')
                text = extract_relevant_text(response.text)
                summary_store[current_url] = f"URL: {current_url}\n\n{text}"

                for link in soup.find_all('a', href=True):
                    new_url = urljoin(current_url, link['href'])
                    new_url = urlparse(new_url)._replace(fragment='').geturl()
                    if urlparse(new_url).netloc == urlparse(url).netloc and new_url not in visited and new_url not in to_visit:
                        to_visit.append(new_url)

                visited.add(current_url)
                pages_crawled += 1

        except Exception as e:
            logging.error(f"Error crawling {current_url}: {str(e)}")

    if progress:
        progress(1.0, f"Crawling complete. Pages crawled: {pages_crawled}")
    
    texts = list(summary_store.values())
    urls = list(summary_store.keys())
    embeddings = get_embeddings(texts)
    
    for text, url, embedding in zip(texts, urls, embeddings):
        if embedding is not None:
            collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"url": url}],
                ids=[url.replace("://", "_").replace("/", "_")]
            )
        
        entities_and_relationships = extract_entities_and_relationships(text, url)
        update_knowledge_graph(entities_and_relationships)
    
    fig = create_plotly_graph()
    
    return "Crawling and Knowledge Graph generation complete.", "Summarization complete.", fig

def query_content(query):
    if client is None:
        return "Error: OpenAI client is not initialized. Please set the API key first.", ""

    query_embedding = get_embeddings([query])[0]
    if query_embedding is None:
        return "Error: Failed to process query. Please try again.", ""
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    context = "\n\n".join(results['documents'][0])
    relevant_urls = [metadata['url'] for metadata in results['metadatas'][0]]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": answer_prompt},
                {"role": "user", "content": f"Query: {query}\nContext:\n{context}"}
            ],
            response_format={"type": "json_object"}
        )
        
        if response.choices[0].finish_reason == "stop":
            result = json.loads(response.choices[0].message.content)
            answer = result.get("answer", "No answer provided.")
            confidence = result.get("confidence", "Unknown")
            additional_info = result.get("additional_info", "")
            
            formatted_answer = f"{answer}\n\nConfidence: {confidence}\n\nAdditional Info: {additional_info}"
            return formatted_answer, ", ".join(relevant_urls)
        else:
            logging.error(f"Unexpected finish reason: {response.choices[0].finish_reason}")
            return "Error: Unexpected response from the model. Please try again.", ""
        
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response: {str(e)}")
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
    
    return "Error: An error occurred while generating the answer. Please try again.", ""

def set_api_key(api_key):
    global client
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        logging.info("API key set and tested successfully.")
        return "API key set and tested successfully."
    except Exception as e:
        logging.error(f"Error setting or testing API key: {str(e)}")
        return f"Failed to set or test API key: {str(e)}"
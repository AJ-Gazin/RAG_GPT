import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from openai import OpenAI
import tiktoken
from readability import Document
import logging
import networkx as nx
import plotly.graph_objects as go
import spacy
import chromadb
from prompts import summarize_prompt, select_urls_prompt, answer_prompt, extract_entities_prompt
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MAX_PAGES = 60
MAX_TOKENS = 8191  # Updated max input tokens for text-embedding-3-small
MAX_OUTPUT_TOKENS = 16000
client = None
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize Chroma client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="website_content")

# Initialize spaCy for NER
nlp = spacy.load("en_core_web_trf")

# Initialize NetworkX graph
G = nx.Graph()

def count_tokens(text):
    """Count the number of tokens in a given text."""
    return len(tokenizer.encode(text))

def sanitize_filename(url):
    """Sanitize URL to create a valid filename."""
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'[\\/*?:"<>|]', '_', url)
    return url[:200]  # Limit length to 200 characters

def prioritize_pages(links):
    """Prioritize links based on keywords."""
    priority_keywords = ['about', 'services', 'products', 'contact', 'team', 'history']
    priority_pages = [link for link in links if any(keyword in link.lower() for keyword in priority_keywords)]
    return priority_pages + [link for link in links if link not in priority_pages]

def extract_relevant_text(html_content):
    """Extract main content from HTML using readability-lxml."""
    doc = Document(html_content)
    return doc.summary()

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding for the given text using OpenAI's API."""
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model, dimensions=1536)
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {str(e)}")
        return None

def extract_entities_and_relationships(text, url):
    """Extract entities and relationships from text using GPT-4o-mini."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": extract_entities_prompt},
                {"role": "user", "content": f"Text: {text}\nURL: {url}"}
            ],
            max_tokens=MAX_OUTPUT_TOKENS
        )
        result = response.choices[0].message.content
        return eval(result)  # Convert string representation of dict to actual dict
    except Exception as e:
        logging.error(f"Error extracting entities and relationships: {str(e)}")
        return {'entities': {}, 'relationships': []}

def update_knowledge_graph(entities_and_relationships):
    """Update the knowledge graph with new entities and relationships."""
    for entity, info in entities_and_relationships['entities'].items():
        G.add_node(entity, **info)
    
    for relation in entities_and_relationships['relationships']:
        G.add_edge(relation['source'], relation['target'], type=relation['type'])

def create_plotly_graph():
    """Create a Plotly graph from the NetworkX graph."""
    pos = nx.spring_layout(G)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{adjacencies[0]}<br># of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Website Knowledge Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    return fig

def crawl_website(url, max_pages, progress=None):
    """Crawl website, extract content, and build knowledge graph."""
    global G
    G = nx.Graph()  # Reset the graph for each new crawl
    
    visited = set()
    to_visit = [url]
    pages_crawled = 0
    total_tokens = 0
    summary_store = {}

    while to_visit and pages_crawled < max_pages and total_tokens < MAX_TOKENS:
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
                text = f"URL: {current_url}\n\n{text}"

                text_tokens = count_tokens(text)
                if total_tokens + text_tokens > MAX_TOKENS:
                    logging.warning("Token limit reached, stopping crawl.")
                    break

                summary_store[current_url] = text
                total_tokens += text_tokens

                new_links = []
                for link in soup.find_all('a', href=True):
                    new_url = urljoin(current_url, link['href'])
                    new_url = urlparse(new_url)._replace(fragment='').geturl()
                    if urlparse(new_url).netloc == urlparse(url).netloc and new_url not in visited and new_url not in to_visit:
                        new_links.append(new_url)

                prioritized_links = prioritize_pages(new_links)
                to_visit.extend(prioritized_links)

                visited.add(current_url)
                pages_crawled += 1

        except Exception as e:
            logging.error(f"Error crawling {current_url}: {str(e)}")

    if progress:
        progress(1.0, f"Crawling complete. Pages crawled: {pages_crawled}")
    
    for current_url, text in summary_store.items():
        # Generate embedding for the text
        embedding = get_embedding(text, model="text-embedding-3-small")
        if embedding is None:
            continue
        
        # Add to Chroma
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"url": current_url}],
            ids=[sanitize_filename(current_url)]
        )
        
        # Extract entities and relationships
        entities_and_relationships = extract_entities_and_relationships(text, current_url)
        update_knowledge_graph(entities_and_relationships)
    
    # Generate the Plotly graph
    fig = create_plotly_graph()
    
    return "Crawling and Knowledge Graph generation complete.", "Summarization complete.", fig

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def query_content(query):
    """Process user query and generate an answer."""
    logging.info("Query processing started.")
    
    # Generate embedding for the query
    query_embedding = get_embedding(query, model="text-embedding-3-small")
    if query_embedding is None:
        return "Error: Failed to process query. Please try again.", ""
    
    # Search for relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    relevant_texts = results['documents'][0]
    relevant_urls = [metadata['url'] for metadata in results['metadatas'][0]]
    
    context = "\n\n".join(relevant_texts)
    
    # Ensure we're within token limit
    while count_tokens(context) + count_tokens(query) + count_tokens(answer_prompt) > MAX_TOKENS:
        context = "\n\n".join(context.split("\n\n")[:-1])  # Remove the last text
    
    # Generate answer
    try:
        answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": answer_prompt},
                {"role": "user", "content": f"Query: {query}\nContext:\n{context}"}
            ],
            max_tokens=MAX_OUTPUT_TOKENS
        ).choices[0].message.content
        logging.info("Answer generated successfully.")
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return "Error: An error occurred while generating the answer. Please check the logs.", ""

    # Add knowledge graph information to the answer
    graph_info = analyze_graph(query)
    answer += f"\n\nKnowledge Graph Insights:\n{graph_info}"
    
    logging.info("Query processing completed.")
    return answer, ", ".join(relevant_urls)

def analyze_graph(query):
    """Analyze the knowledge graph based on the query."""
    info = []
    info.append(f"Total entities: {G.number_of_nodes()}")
    info.append(f"Total relationships: {G.number_of_edges()}")
    
    # Find most connected entities
    degree_centrality = nx.degree_centrality(G)
    top_entities = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:5]
    info.append("Top connected entities:")
    for entity in top_entities:
        info.append(f"- {entity}")
    
    # Find entities most relevant to the query
    query_doc = nlp(query)
    query_entities = [ent.text for ent in query_doc.ents]
    relevant_entities = [node for node in G.nodes() if any(entity.lower() in node.lower() for entity in query_entities)]
    if relevant_entities:
        info.append("Query-relevant entities:")
        for entity in relevant_entities[:5]:
            info.append(f"- {entity}")
    
    return "\n".join(info)

def set_api_key(api_key):
    """Set and test the OpenAI API key."""
    global client
    try:
        client = OpenAI(api_key=api_key)
        # Test the API key
        client.models.list()
        logging.info("API key set and tested successfully.")
        return "API key set and tested successfully."
    except Exception as e:
        logging.error(f"Error setting or testing API key: {str(e)}")
        return f"Failed to set or test API key: {str(e)}"
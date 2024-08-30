import asyncio
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
#from openai import OpenAI
import tiktoken
from datetime import datetime
import logging
import networkx as nx
from graspologic.partition import hierarchical_leiden
from collections import defaultdict
from pyvis.network import Network
from typing import List, Callable, Any, Optional, Union

from llama_index.core import (
    PropertyGraphIndex,
    Document,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.llms.llm import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms import ChatMessage

from dotenv import load_dotenv
from prompts import answer_prompt, KG_TRIPLET_EXTRACT_TMPL

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MAX_TOKENS = 128000
MAX_OUTPUT_TOKENS = 16000
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tokenizer = tiktoken.get_encoding("cl100k_base")
content_store = {}

# Configure global settings
Settings.llm = LlamaOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
Settings.chunk_size = 4096
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
        logging.info(f"Crawling page: {current_url}")
        response = requests.get(current_url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
            text = f"URL: {current_url}\n\n{extract_text_from_html(response.text)}"
            text_tokens = count_tokens(text)
            if total_tokens + text_tokens > max_tokens:
                logging.warning("Token limit reached, stopping crawl.")
                return None, total_tokens
            content_store[current_url] = text
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
    global content_store
    content_store.clear()
    
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

entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'

def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    
    valid_entities = []
    for e in entities:
        if len(e) == 3:
            valid_entities.append(e)
        else:
            logging.error(f"Invalid entity format: {e}")
    
    valid_relationships = []
    for r in relationships:
        if len(r) == 4:
            valid_relationships.append(r)
        else:
            logging.error(f"Invalid relationship format: {r}")
    
    return valid_entities, valid_relationships

class GraphRAGExtractor(TransformComponent):
    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = parse_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or KG_TRIPLET_EXTRACT_TMPL,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError as e:
            logging.error(f"Error parsing LLM response: {e}")
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for subj, obj, rel, description in entities_relationship:
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    entity_info = None
    max_cluster_size = 5

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = Settings.llm.chat(messages)
        return response.message.content.strip()

    def build_communities(self):
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            description = relation.properties.get("relationship_description", "")
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=description,
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."
            self.community_summary[community_id] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20

    def custom_query(self, query_str: str) -> str:
        logging.info(f"Processing query: {query_str}")
        entities = self.get_entities(query_str, self.similarity_top_k)
        community_ids = self.retrieve_entity_communities(self.graph_store.entity_info, entities)
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ]
        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def get_entities(self, query_str, similarity_top_k):
        logging.info(f"Retrieving entities for query: {query_str}")
        nodes_retrieved = self.index.as_retriever(similarity_top_k=similarity_top_k).retrieve(query_str)
        entities = set()
        pattern = r"(\w+(?:\s+\w+)*)\s*\({[^}]*}\)\s*->\s*([^(]+?)\s*\({[^}]*}\)\s*->\s*(\w+(?:\s+\w+)*)"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.DOTALL)
            for match in matches:
                subject = match[0]
                obj = match[2]
                entities.add(subject)
                entities.add(obj)

        return list(entities)
    def retrieve_entity_communities(self, entity_info, entities):
        logging.info(f"Retrieving communities for entities: {entities}")
        community_ids = []
        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])
        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary, query):
        logging.info("Generating answer from community summary")
        prompt = f"Given the community summary: {community_summary}, how would you answer the following query? Query: {query}"
        response = self.llm.complete(prompt)
        return response.text.strip()

    def aggregate_answers(self, community_answers):
        logging.info("Aggregating answers from communities")
        prompt = "Combine the following intermediate answers into a final, concise response."
        response = self.llm.complete(prompt)
        return response.text.strip()

def analyze_website():
    logging.info("Starting website analysis")
    
    splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents([Document(text=content) for content in content_store.values()])
    
    kg_extractor = GraphRAGExtractor(
        llm=Settings.llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=2,
        parse_fn=parse_fn
    )
    
    graph_store = GraphRAGStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI")
    )
    
    extracted_nodes = kg_extractor(nodes)
    
    try:
        index = PropertyGraphIndex(
            nodes=extracted_nodes,
            property_graph_store=graph_store,
            storage_context=StorageContext.from_defaults(graph_store=graph_store),
            show_progress=True,
        )
        
        # Persist the index
        output_dir = os.getenv("EMBEDDING_DIR", "embeddings")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        embedding_path = f"{output_dir}/{timestamp}"
        index.storage_context.persist(persist_dir=embedding_path)
        
    except Exception as e:
        logging.error(f"Error building PropertyGraphIndex: {str(e)}")
        return "Analysis failed: Error building knowledge graph.", ""

    try:
        graph_store.build_communities()
    except Exception as e:
        logging.error(f"Error building communities: {str(e)}")
        return "Analysis complete, but community building failed.", ""

    try:
        graph_html = generate_graph_visualization(graph_store)
        with open("graph.html", "w") as f:
            f.write(graph_html)
    except Exception as e:
        logging.error(f"Error generating graph visualization: {str(e)}")
        graph_html = ""

    logging.info("Website analysis complete")
    return "Analysis complete.", graph_html

def generate_graph_visualization(graph_store):
    logging.info("Generating graph visualization")
    g = graph_store._create_nx_graph()
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(g)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)
    return net.generate_html()

def query_content(query):
    logging.info(f"Processing query: {query}")
    vector_dir = os.getenv("EMBEDDING_DIR", "embeddings")
    latest_vector_dir = get_latest_dir(vector_dir)
    if not latest_vector_dir:
        logging.error("No property graph index found")
        return "Error: No data available to query.", ""
    graph_store = GraphRAGStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI")
    )
    
    property_graph_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=latest_vector_dir))
    
    query_engine = GraphRAGQueryEngine(
        graph_store=graph_store,
        index=property_graph_index,
        llm=Settings.llm,
        similarity_top_k=10,
    )

    try:
        response = query_engine.custom_query(query)
    except Exception as e:
        logging.error(f"Error during query processing: {str(e)}")
        return f"Error occurred while processing the query: {str(e)}", ""

    urls = list(content_store.keys())

    return response, ", ".join(urls)

def get_latest_dir(parent_dir):
    logging.info(f"Getting latest directory in {parent_dir}")
    if not os.path.exists(parent_dir):
        logging.warning(f"Directory {parent_dir} does not exist")
        return None
    dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    return max(dirs, key=os.path.getmtime) if dirs else None

if __name__ == "__main__":
    # You can add any testing or standalone functionality here
    pass

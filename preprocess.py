##interesting note: chunk 4096 = ~1 node per document. 2048 = ~1.3 nodes per document. 

import os
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import networkx as nx
from pyvis.network import Network
import asyncio
from typing import List, Callable, Any, Optional, Union
from datetime import datetime
import logging
from collections import defaultdict
from prompts import KG_TRIPLET_EXTRACT_TMPL
from llama_index.core import (
    PropertyGraphIndex,
    Document,
    StorageContext,
    Settings,
)
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.llms.llm import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms import ChatMessage

from graspologic.partition import hierarchical_leiden

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure global settings
Settings.llm = LlamaOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
Settings.chunk_size = 4096
Settings.chunk_overlap = 20

# Constants
MAX_TOKENS = 128000
MAX_OUTPUT_TOKENS = 16000


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def crawl_directory(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() == 'index.html':
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                url_path = '/'.join(relative_path.split(os.sep)[:-1])  # Remove 'index.html' from the path
                if not url_path:
                    url_path = '/'  # Root index.html
                else:
                    url_path = '/' + url_path + '/'
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    text = extract_text_from_html(html_content)
                    
                    # Create a Document with the extracted text and metadata
                    doc = Document(
                        text=text,
                        metadata={
                            "source": url_path,
                            "filename": file,
                            "filepath": file_path
                        }
                    )
                    documents.append(doc)
    
    return documents

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
        if len(nx_graph) == 0:
            logging.warning("Graph is empty. No communities to build.")
            return

        try:
            community_hierarchical_clusters = hierarchical_leiden(
                nx_graph, max_cluster_size=self.max_cluster_size
            )
            self.entity_info, community_info = self._collect_community_info(
                nx_graph, community_hierarchical_clusters
            )
            self._summarize_communities(community_info)
        except Exception as e:
            logging.error(f"Error in build_communities: {str(e)}")
            raise
    def _create_nx_graph(self):
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            # Ensure node names are strings and don't include 'True' or numeric values
            node1 = str(entity1.name).replace('True', 'true').replace('False', 'false')
            node2 = str(entity2.name).replace('True', 'true').replace('False', 'false')
            if node1.isdigit():
                node1 = f"node_{node1}"
            if node2.isdigit():
                node2 = f"node_{node2}"
            
            nx_graph.add_node(node1, type=entity1.label)
            nx_graph.add_node(node2, type=entity2.label)
            description = relation.properties.get("relationship_description", "")
            nx_graph.add_edge(
                node1,
                node2,
                relationship=relation.label,
                description=description,
            )
        logging.info(f"Created graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
        return nx_graph

    def build_communities(self):
        nx_graph = self._create_nx_graph()
        if len(nx_graph) == 0:
            logging.warning("Graph is empty. No communities to build.")
            return

        try:
            logging.info(f"Starting community detection with {len(nx_graph)} nodes")
            community_hierarchical_clusters = hierarchical_leiden(
                nx_graph, max_cluster_size=self.max_cluster_size
            )
            logging.info(f"Community detection completed. Number of clusters: {len(community_hierarchical_clusters)}")
            self.entity_info, community_info = self._collect_community_info(
                nx_graph, community_hierarchical_clusters
            )
            self._summarize_communities(community_info)
        except Exception as e:
            logging.error(f"Error in build_communities: {str(e)}")
            raise

    def _collect_community_info(self, nx_graph, clusters):
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        logging.info(f"Collecting community info for {len(clusters)} clusters")

        for cluster in clusters:
            node = cluster.node
            cluster_id = cluster.cluster
            if node in nx_graph:
                entity_info[node].add(cluster_id)
                for neighbor in nx_graph.neighbors(node):
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
            else:
                logging.warning(f"Node '{node}' from cluster {cluster_id} not found in graph.")

        entity_info = {k: list(v) for k, v in entity_info.items()}
        logging.info(f"Collected info for {len(entity_info)} entities across {len(community_info)} communities")
        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."
            self.community_summary[community_id] = self.generate_community_summary(details_text)
        logging.info(f"Generated summaries for {len(self.community_summary)} communities")

    def get_community_summaries(self):
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

def generate_and_save_graph_visualization(graph_store, output_file):
    logging.info("Generating graph visualization")
    g = graph_store._create_nx_graph()
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(g)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)
    net.save_graph(output_file)
    logging.info(f"Graph visualization saved to {output_file}")

def preprocess_neuronslab():
    logging.info("Starting preprocessing of NeuronsLab.com")
    
    neuronslab_dir = "NeuronsLab.com"  # Replace with actual path
    documents = crawl_directory(neuronslab_dir)
    logging.info(f"Crawled {len(documents)} documents from NeuronsLab.com")
    
    splitter = SentenceSplitter(chunk_size=4096, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    logging.info(f"Created {len(nodes)} nodes from documents")
    
    kg_extractor = GraphRAGExtractor(
        llm=Settings.llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=1,
        parse_fn=parse_fn
    )
    
    extracted_nodes = kg_extractor(nodes)
    logging.info(f"Extracted knowledge graph from {len(extracted_nodes)} nodes")
    
    graph_store = GraphRAGStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI")
    )
    
    try:
        index = PropertyGraphIndex(
            nodes=extracted_nodes,
            property_graph_store=graph_store,
            storage_context=StorageContext.from_defaults(graph_store=graph_store),
            show_progress=True,
        )
        
        output_dir = "neuronslab_example_embeddings"
        os.makedirs(output_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=output_dir)
        
        logging.info(f"Index persisted to {output_dir}")
        
        # Log information about the graph structure
        nx_graph = graph_store._create_nx_graph()
        logging.info(f"Created graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
        logging.info(f"Node types: {set(nx.get_node_attributes(nx_graph, 'type').values())}")
        logging.info(f"Edge types: {set(nx.get_edge_attributes(nx_graph, 'relationship').values())}")
        
        graph_store.build_communities()
        logging.info("Communities built successfully")
        
        # Log information about the communities
        community_summaries = graph_store.get_community_summaries()
        logging.info(f"Number of communities: {len(community_summaries)}")
        logging.info(f"Community sizes: {[len(comm.split('.')) for comm in community_summaries.values()]}")
        
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        return
    
    try:
        visualization_file = os.path.join(output_dir, "neuronslab_graph_visualization.html")
        generate_and_save_graph_visualization(graph_store, visualization_file)
        logging.info(f"Graph visualization saved to {visualization_file}")
    except Exception as e:
        logging.error(f"Error generating graph visualization: {str(e)}")
    
    logging.info("Preprocessing of NeuronsLab.com completed")
if __name__ == "__main__":
    try:
        preprocess_neuronslab()
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {str(e)}")
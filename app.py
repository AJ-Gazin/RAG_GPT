# implement: https://llm-graph-builder.neo4jlabs.com/
import os
import gradio as gr
import logging
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from theme import BusinessAnalyzerTheme  # Custom theme


# Load environment variables from .env file

load_dotenv()


# Retrieve the OpenAI API token and Neo4j credentials from the environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NEO4J_URL = os.getenv("NEO4J_URI")


if not OPENAI_API_KEY or not NEO4J_USERNAME or not NEO4J_PASSWORD or not NEO4J_URL:
    raise ValueError(
        "Required environment variables (OpenAI API token, Neo4j credentials) are missing."
    )


# Set up logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Neo4j Setup

graph_store = Neo4jPropertyGraphStore(
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    url=NEO4J_URL,
)


# Initialize OpenAI LLM via LlamaIndex

llm = OpenAILLM(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.0)

embedding_model = OpenAIEmbedding(model_name="text-embedding-3-small")


# Global variable to store the loaded index

loaded_index = None


def load_and_process_html():
    """Load HTML files from NeuronsLab.com folder and process them into a knowledge graph."""

    global loaded_index

    logging.info("Loading and processing HTML files from NeuronsLab.com folder.")

    kg_extractor = SimpleLLMPathExtractor(
        llm=llm,
        max_paths_per_chunk=10,
        num_workers=4,
    )

    try:
        # Assuming the HTML files are in a folder named 'NeuronsLab.com'

        documents = SimpleDirectoryReader("./NeuronsLabParsed/").load_data()

        # Create the PropertyGraphIndex

        # loaded_index = PropertyGraphIndex.from_documents(
        #     documents,
        #     embed_model=embedding_model,
        #     kg_extractors=[kg_extractor],
        #     property_graph_store=graph_store,
        #     show_progress=True,
        # )

        loaded_index = PropertyGraphIndex.from_existing(
            llm=llm,
            embed_model=embedding_model,
            property_graph_store=graph_store,
            show_progress=True,
        )

        return "Successfully loaded and processed HTML files into a knowledge graph."

    except Exception as e:
        logging.error(f"Error loading and processing HTML files: {str(e)}")

        return f"Error loading and processing HTML files: {str(e)}"


def query_content(query):
    """Process a query and return results from the Neo4j-backed property graph."""

    logging.info("Query processing started.")

    global loaded_index

    if loaded_index is None:
        logging.error("Error: Property graph index not loaded.")

        return (
            "Error: Property graph index not loaded. Please load the graph first.",
            "",
        )

    try:
        # Retrieve nodes from the property graph

        retriever = loaded_index.as_retriever(include_text=False)

        nodes = retriever.retrieve(query)

        node_texts = [node.text for node in nodes]

        if not node_texts:
            return "No nodes found for the query.", ""

        logging.info(f"Nodes retrieved: {node_texts}")

        # Generate a summary using the query engine

        query_engine = loaded_index.as_query_engine(include_text=True)

        response = query_engine.query(query)

        logging.info("Answer generated successfully.")

        return str(response), "\n".join(node_texts)

    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")

        return f"Error: {str(e)}", ""


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        css=".center { text-align: center; width: 100%; } .radio-button { flex: 1; text-align: center; }",
        theme=BusinessAnalyzerTheme(),
    ) as demo:
        gr.Markdown("# NeuronsLab.com Knowledge Graph Analyzer", elem_classes="center")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Load and Process HTML Files", elem_classes="center")

                load_button = gr.Button(
                    "Load and Process HTML Files", variant="primary"
                )

                gr.Markdown("## Content Query", elem_classes="center")

                query_input = gr.Textbox(
                    label="Enter Your Query",
                    show_label=False,
                    placeholder="Type Your Query",
                    elem_classes="center",
                )

                query_button = gr.Button("Ask", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("## Load Results", elem_classes="center")

                load_output = gr.Textbox(
                    label="Load Status", show_label=False, placeholder="Load Status"
                )

                gr.Markdown("## Query Results", elem_classes="center")

                answer_output = gr.Markdown()

                urls_output = gr.Textbox(
                    label="Sources", show_label=False, placeholder="Sources"
                )

        # Event handlers

        load_button.click(fn=load_and_process_html, outputs=[load_output])

        query_button.click(
            fn=query_content, inputs=query_input, outputs=[answer_output, urls_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()

    demo.launch(share=True)

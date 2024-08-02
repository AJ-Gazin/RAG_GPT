import os
import gradio as gr
from firecrawl import FirecrawlApp
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from neo4j import GraphDatabase
import spacy
import openai

# Load environment variables
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Initialize components
app = FirecrawlApp(api_key=firecrawl_api_key)
llm = OpenAI(temperature=0.7, api_key=openai_api_key)
nlp = spacy.load("en_core_web_sm")
openai.api_key = openai_api_key

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_entity(self, entity, urls):
        with self.driver.session() as session:
            session.execute_write(self._create_entity, entity, urls)

    @staticmethod
    def _create_entity(tx, entity, urls):
        query = (
            "MERGE (e:Entity {name: $name}) "
            "ON CREATE SET e.urls = $urls "
            "ON MATCH SET e.urls = e.urls + $urls"
        )
        tx.run(query, name=entity, urls=urls)

    def add_relationship(self, entity1, entity2, relationship):
        with self.driver.session() as session:
            session.execute_write(self._create_relationship, entity1, entity2, relationship)

    @staticmethod
    def _create_relationship(tx, entity1, entity2, relationship):
        query = (
            "MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2}) "
            "MERGE (e1)-[r:RELATED {type: $relationship}]->(e2)"
        )
        tx.run(query, entity1=entity1, entity2=entity2, relationship=relationship)

    def get_relevant_urls(self, query):
        with self.driver.session() as session:
            return session.execute_read(self._get_relevant_urls, query)

    @staticmethod
    def _get_relevant_urls(tx, query):
        cypher_query = """
        MATCH (e:Entity)
        WHERE any(term IN $query_terms WHERE e.name CONTAINS term)
        MATCH (e)-[*0..1]-(related:Entity)
        RETURN DISTINCT related.urls as urls
        """
        result = tx.run(cypher_query, query_terms=query.lower().split())
        return list(set([url for record in result for url in record['urls']]))

    def clear_database(self):
        with self.driver.session() as session:
            session.execute_write(self._clear_database)

    @staticmethod
    def _clear_database(tx):
        tx.run("MATCH (n) DETACH DELETE n")

neo4j_conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)

def create_knowledge_graph(content, url):
    doc = nlp(content)
    entities = {}

    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            normalized = token.lemma_.lower()
            if normalized not in entities:
                entities[normalized] = set()
            entities[normalized].add(url)

    for entity, urls in entities.items():
        neo4j_conn.add_entity(entity, list(urls))

    for token in doc:
        if token.dep_ in ['nsubj', 'dobj'] and token.head.pos_ in ['NOUN', 'PROPN']:
            subject = token.lemma_.lower()
            object = token.head.lemma_.lower()
            if subject in entities and object in entities:
                neo4j_conn.add_relationship(subject, object, token.dep_)

def crawl_website(url):
    try:
        # Clear the existing database before crawling
        neo4j_conn.clear_database()

        crawl_result = app.crawl_url(url, {
            'crawlerOptions': {
                'maxDepth': 2,
                'limit': 20
            },
            'pageOptions': {
                'onlyMainContent': True
            }
        })

        content_store = {}
        for result in crawl_result:
            content = result.get('content', '')
            url = result['metadata']['sourceURL']
            create_knowledge_graph(content, url)
            content_store[url] = content

        return content_store

    except Exception as e:
        return {"error": f"An error occurred during crawling: {str(e)}"}

def answer_query(query, context):
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="Given the query: {query}\nAnd the following context:\n{context}\nProvide a comprehensive answer to the query."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query=query, context=context)

def process_website_and_query(url, query):
    gr.Info("Crawling the website. This may take 10-15 seconds...")
    content_store = crawl_website(url)
    
    if isinstance(content_store, dict) and "error" in content_store:
        return content_store["error"]
    
    if not query:
        return "Website crawled successfully. Please enter a query to get an answer."
    
    relevant_urls = neo4j_conn.get_relevant_urls(query)
    context = "\n".join([f"URL: {url}\nContent: {content_store.get(url, '')}" for url in relevant_urls])
    answer = answer_query(query, context)
    
    return answer

# Gradio interface
iface = gr.Interface(
    fn=process_website_and_query,
    inputs=[
        gr.Textbox(label="Website URL"),
        gr.Textbox(label="Your Query (optional)")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Website Query Assistant",
    description="Enter a website URL to crawl. Optionally, you can also enter a query to get an answer based on the crawled content.",
)

if __name__ == "__main__":
    iface.launch()
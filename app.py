# app.py

import gradio as gr
from firecrawl import FirecrawlApp
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from prompts import summarize_prompt, select_urls_prompt, answer_prompt  # Import prompts

# Load environment variables from a .env file
load_dotenv()

# Initialize Firecrawl and OpenAI
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

app = FirecrawlApp(api_key=firecrawl_api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_api_key)

# Directory to store crawled pages
#TODO: replace with proper database
os.makedirs("parsed_pages", exist_ok=True)

# Dictionary to store summaries
summary_store = {}

def crawl_and_store(url):
    try:
        # Start a crawl
        crawl_results = app.crawl_url(url, {
            'crawlerOptions': {
                'maxDepth': 2,
                'limit': 25,
                'onlyMainContent': True
            },
            'wait_until_done': True
        })

        # Ensure crawl_results is a list
        if not isinstance(crawl_results, list):
            return "Unexpected response from crawl job. Please check the URL and try again."

        # Store markdown content in files and create summaries
        markdown_contents = []
        for result in crawl_results:
            markdown = result.get('markdown', '')
            source_url = result['metadata'].get('sourceURL', 'unknown_url')
            sanitized_filename = source_url.replace('https://', '').replace('http://', '').replace('/', '_')
            file_path = os.path.join("parsed_pages", f"{sanitized_filename}.md")
            
            # Write markdown to file
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(markdown)
            
            # Summarize content
            summary = summarize_content(markdown)
            summary_store[sanitized_filename] = summary
            markdown_contents.append(f"Stored: {file_path} with summary.")

        # Return message indicating stored files
        return "\n".join(markdown_contents)

    except Exception as e:
        return f"An error occurred during crawling: {str(e)}"

def summarize_content(content):
    try:
        # Use the imported summarize_prompt
        chain = summarize_prompt | llm | StrOutputParser()
        summary = chain.invoke({"content": content}).strip()
        return summary
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"

def select_relevant_urls(query):
    try:
        # Prepare the summaries for the selection prompt
        summaries = "\n".join([f"{url}: {summary}" for url, summary in summary_store.items()])
        
        # Use the imported select_urls_prompt
        chain = select_urls_prompt | llm | StrOutputParser()
        selected_urls = chain.invoke({"query": query, "summaries": summaries}).strip()
        return selected_urls.split(",") if selected_urls else []

    except Exception as e:
        return []

def answer_query(query):
    try:
        selected_urls = select_relevant_urls(query)
        if not selected_urls:
            disclaimer = "No relevant URLs were found."
        else:
            disclaimer = ""

        # Read content from selected URLs
        context_parts = []
        for url in selected_urls:
            sanitized_filename = url.strip().replace('https://', '').replace('http://', '').replace('/', '_')
            file_path = os.path.join("parsed_pages", f"{sanitized_filename}.md")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    context_parts.append(file.read())

        # Compile the context from selected markdown content
        context = "\n\n".join(context_parts)
        
        # Use the imported answer_prompt
        chain = answer_prompt | llm | StrOutputParser()
        answer = chain.invoke({"query": query, "context": context})

        # Format selected URLs as a string
        urls_output = ", ".join(selected_urls) if selected_urls else "No URLs selected."

        return disclaimer + answer, urls_output

    except Exception as e:
        return f"An error occurred while answering the query: {str(e)}", "Error retrieving URLs"

# Gradio interfaces for crawling and querying
crawl_interface = gr.Interface(
    fn=crawl_and_store,
    inputs=gr.Textbox(label="Website URL"),
    outputs=gr.Textbox(label="Stored Markdown Files and Summaries"),
    title="Website Crawler",
    description="Enter a website URL to crawl with a depth of 1 and store the markdown content in files and summaries.",
)

query_interface = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(label="Your Query"),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Relevant URLs"),
    ],
    title="Query Assistant",
    description="Enter a query to get an answer based on the crawled content stored in files. The relevant URLs used as context will also be shown.",
)

# Combine interfaces into a tabbed interface
iface = gr.TabbedInterface([crawl_interface, query_interface], ["Crawl Website", "Query Content"])
iface.launch()

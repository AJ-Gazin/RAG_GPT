# app.py

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from prompts import summarize_prompt, select_urls_prompt, answer_prompt
import requests
from bs4 import BeautifulSoup
from lxml import html
from urllib.parse import urljoin, urlparse
import html2text
import re
import hashlib

# Load environment variables from a .env file
load_dotenv()

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_api_key)

# Directory to store crawled pages
os.makedirs("parsed_pages", exist_ok=True)

# Dictionary to store summaries
summary_store = {}

# Initialize html2text
h = html2text.HTML2Text()
h.ignore_links = False
h.ignore_images = False
h.ignore_emphasis = False
h.body_width = 0  # Don't wrap text

def sanitize_filename(url):
    # Remove the protocol (http:// or https://)
    url = re.sub(r'^https?://', '', url)
    
    # Replace special characters with underscore
    url = re.sub(r'[\\/*?:"<>|]', '_', url)
    
    # Limit the length of the filename
    max_length = 200  # Maximum length for the base filename
    if len(url) > max_length:
        # If the URL is too long, use a part of it and add a hash
        hash_object = hashlib.md5(url.encode())
        url_hash = hash_object.hexdigest()[:10]  # Use first 10 characters of the hash
        url = url[:max_length-11] + '_' + url_hash
    
    return url + '.md'

def crawl_url(url, max_depth=2, max_size_mb=10, progress=gr.Progress()):
    visited = set()
    to_visit = [(url, 0)]
    results = []
    total_size_bytes = 0
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes

    progress(0, desc="Starting crawl...")

    while to_visit and total_size_bytes < max_size_bytes:
        current_url, depth = to_visit.pop(0)
        if current_url in visited or depth > max_depth:
            continue

        visited.add(current_url)

        try:
            progress(len(visited) / (len(visited) + len(to_visit)), desc=f"Crawling: {current_url}")
            
            response = requests.get(current_url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'lxml')
                tree = html.fromstring(response.content)

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Convert to markdown
                markdown = html_to_markdown(str(soup))
                
                # Calculate size of markdown content
                content_size = len(markdown.encode('utf-8'))
                
                # Check if adding this content would exceed the size limit
                if total_size_bytes + content_size > max_size_bytes:
                    progress(1, desc="Size limit reached. Stopping crawl.")
                    break

                total_size_bytes += content_size

                results.append({
                    'markdown': markdown,
                    'metadata': {'sourceURL': current_url, 'size_bytes': content_size}
                })

                if depth < max_depth:
                    links = tree.xpath('//a/@href')
                    for link in links:
                        absolute_url = urljoin(current_url, link)
                        if urlparse(absolute_url).netloc == urlparse(url).netloc:
                            to_visit.append((absolute_url, depth + 1))

        except Exception as e:
            progress(len(visited) / (len(visited) + len(to_visit)), desc=f"Error crawling {current_url}: {str(e)}")

    progress(1, desc=f"Crawl complete. Total size: {total_size_bytes / (1024 * 1024):.2f} MB")
    return results

def html_to_markdown(html_content):
    return h.handle(html_content)

def crawl_and_store(url, max_size_mb=10, progress=gr.Progress()):
    try:
        crawl_results = crawl_url(url, max_size_mb=max_size_mb, progress=progress)

        markdown_contents = []
        total_stored_size = 0
        
        progress(0, desc="Processing crawled content...")
        
        for i, result in enumerate(crawl_results):
            markdown = result['markdown']
            source_url = result['metadata']['sourceURL']
            content_size = result['metadata']['size_bytes']
            sanitized_filename = sanitize_filename(source_url)
            file_path = os.path.join("parsed_pages", sanitized_filename)
            
            # Write markdown to file
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(markdown)
            
            # Summarize content
            summary = summarize_content(markdown)
            summary_store[sanitized_filename] = summary
            markdown_contents.append(f"Stored: {file_path} with summary. Size: {content_size / 1024:.2f} KB")
            total_stored_size += content_size
            
            progress((i + 1) / len(crawl_results), desc=f"Processed {i + 1}/{len(crawl_results)} pages")

        return f"Total stored size: {total_stored_size / (1024 * 1024):.2f} MB\n" + "\n".join(markdown_contents)

    except Exception as e:
        return f"An error occurred during crawling: {str(e)}"

def summarize_content(content):
    try:
        chain = summarize_prompt | llm | StrOutputParser()
        summary = chain.invoke({"content": content}).strip()
        return summary
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"

def select_relevant_urls(query):
    try:
        summaries = "\n".join([f"{url}: {summary}" for url, summary in summary_store.items()])
        chain = select_urls_prompt | llm | StrOutputParser()
        selected_urls = chain.invoke({"query": query, "summaries": summaries}).strip()
        return selected_urls.split(",") if selected_urls else []
    except Exception as e:
        return []

def answer_query(query):
    try:
        selected_urls = select_relevant_urls(query)
        if not selected_urls:
            disclaimer = "No relevant URLs were found. "
        else:
            disclaimer = ""

        context_parts = []
        for url in selected_urls:
            sanitized_filename = sanitize_filename(url.strip())
            file_path = os.path.join("parsed_pages", sanitized_filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    context_parts.append(file.read())

        context = "\n\n".join(context_parts)
        
        chain = answer_prompt | llm | StrOutputParser()
        answer = chain.invoke({"query": query, "context": context})

        urls_output = ", ".join(selected_urls) if selected_urls else "No URLs selected."

        return disclaimer + answer, urls_output

    except Exception as e:
        return f"An error occurred while answering the query: {str(e)}", "Error retrieving URLs"

# Gradio interfaces for crawling and querying
crawl_interface = gr.Interface(
    fn=crawl_and_store,
    inputs=[
        gr.Textbox(label="Website URL"),
        gr.Number(label="Max Size (MB)", value=10)
    ],
    outputs=gr.Textbox(label="Stored Markdown Files and Summaries"),
    title="Website Crawler",
    description="Enter a website URL to crawl. The crawler will stop when the total size of crawled content reaches the specified limit.",
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

if __name__ == "__main__":
    iface.launch()
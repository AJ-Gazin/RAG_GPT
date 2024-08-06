# app.py

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from prompts import summarize_prompt, select_urls_prompt, answer_prompt
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from lxml import html
from urllib.parse import urljoin, urlparse
import html2text
import re
import hashlib
import markdown2
import json

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

# Global in-memory storage
in_memory_storage = {}
in_memory_storage_size = 0
MAX_MEMORY_SIZE = 150 * 1024 * 1024  # 150 MB in bytes

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

def html_to_markdown(html_content):
    return h.handle(html_content)

def save_summaries_to_disk():
    with open("summaries.json", "w") as f:
        json.dump(summary_store, f)

def load_summaries_from_disk():
    global summary_store
    try:
        with open("summaries.json", "r") as f:
            summary_store = json.load(f)
    except FileNotFoundError:
        summary_store = {}

async def crawl_url(url, max_depth=2, max_size_mb=10, max_concurrency=5, crawl_progress=gr.Progress()):
    global in_memory_storage, in_memory_storage_size
    visited = set()
    to_visit = [(url, 0)]
    results = []
    total_size_bytes = 0
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    semaphore = asyncio.Semaphore(max_concurrency)

    crawl_progress(0, desc="Starting crawl...")
    total_urls = 1  # Start with 1 for the initial URL

    async with aiohttp.ClientSession() as session:
        while to_visit and total_size_bytes < max_size_bytes:
            current_url, depth = to_visit.pop(0)
            if current_url in visited or depth > max_depth:
                continue

            visited.add(current_url)
            sanitized_filename = sanitize_filename(current_url)
            file_path = os.path.join("parsed_pages", sanitized_filename)

            try:
                crawl_progress(len(visited) / total_urls, desc=f"Processing: {current_url}")
                
                # Check if the URL is already processed
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        markdown = file.read()
                    source = "disk"
                    content_size = len(markdown.encode('utf-8'))
                else:
                    # Fetch and process new content
                    async with semaphore:
                        async with session.get(current_url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                            if response.status == 200:
                                content = await response.text()
                                soup = BeautifulSoup(content, 'lxml')
                                for script in soup(["script", "style"]):
                                    script.decompose()
                                markdown = html_to_markdown(str(soup))
                                content_size = len(markdown.encode('utf-8'))
                                source = "network"

                                # Store the new content
                                with open(file_path, "w", encoding="utf-8") as file:
                                    file.write(markdown)

                                # Look for new links
                                tree = html.fromstring(content)
                                links = tree.xpath('//a/@href')
                                for link in links:
                                    absolute_url = urljoin(current_url, link)
                                    if urlparse(absolute_url).netloc == urlparse(url).netloc:
                                        to_visit.append((absolute_url, depth + 1))
                                        total_urls += 1

                # Check if adding this content would exceed the size limit
                if total_size_bytes + content_size > max_size_bytes:
                    crawl_progress(1, desc="Size limit reached. Stopping crawl.")
                    break

                total_size_bytes += content_size

                results.append({
                    'markdown': markdown,
                    'metadata': {'sourceURL': current_url, 'size_bytes': content_size, 'source': source}
                })

            except Exception as e:
                crawl_progress(len(visited) / total_urls, desc=f"Error processing {current_url}: {str(e)}")

    crawl_progress(1, desc=f"Crawl complete. Total size: {total_size_bytes / (1024 * 1024):.2f} MB")
    return results

def crawl_and_store(url, max_size_mb=10, crawl_progress=gr.Progress()):
    global in_memory_storage, in_memory_storage_size
    try:
        load_summaries_from_disk()
        crawl_results = asyncio.run(crawl_url(url, max_size_mb=max_size_mb, crawl_progress=crawl_progress))
        
        # Process results
        markdown_contents = []
        total_stored_size = 0
        
        for i, result in enumerate(crawl_results):
            markdown = result['markdown']
            source_url = result['metadata']['sourceURL']
            content_size = result['metadata']['size_bytes']
            source = result['metadata']['source']
            
            sanitized_filename = sanitize_filename(source_url)
            
            # Summarize content if not already summarized
            if sanitized_filename not in summary_store:
                summary = summarize_content(markdown)
                summary_store[sanitized_filename] = summary
            
            markdown_contents.append(f"Processed: {source_url} (from {source}). Size: {content_size / 1024:.2f} KB")
            total_stored_size += content_size
            
            crawl_progress((i + 1) / len(crawl_results), desc=f"Processed {i + 1}/{len(crawl_results)} pages")

        # Save summaries to disk
        save_summaries_to_disk()

        # Clear in-memory storage after processing
        in_memory_storage.clear()
        in_memory_storage_size = 0

        output = f"Total processed size: {total_stored_size / (1024 * 1024):.2f} MB\n" + "\n".join(markdown_contents)
        return output

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

        # Convert markdown to HTML
        answer_html = markdown2.markdown(answer)

        urls_output = ", ".join(selected_urls) if selected_urls else "No URLs selected."

        return disclaimer + answer_html, urls_output

    except Exception as e:
        return f"An error occurred while answering the query: {str(e)}", "Error retrieving URLs"

def clear_in_memory_storage():
    global in_memory_storage, in_memory_storage_size
    in_memory_storage.clear()
    in_memory_storage_size = 0

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Website Crawler and Query Assistant")
        
        with gr.Tab("Crawl Website"):
            with gr.Row():
                url_input = gr.Textbox(label="Website URL", scale=4)
                max_size_input = gr.Number(label="Max Size (MB)", value=10, scale=1)
            crawl_output = gr.Textbox(label="Stored Markdown Files and Summaries")
            crawl_button = gr.Button("Crawl")
            crawl_progress = gr.Progress()
            crawl_button.click(
                fn=crawl_and_store, 
                inputs=[url_input, max_size_input], 
                outputs=crawl_output,
                show_progress=crawl_progress
            )
        
        with gr.Tab("Query Content"):
            query_input = gr.Textbox(label="Your Query")
            answer_output = gr.HTML(label="Answer")
            urls_output = gr.Textbox(label="Relevant URLs")
            query_button = gr.Button("Query")
            query_button.click(fn=answer_query, inputs=query_input, outputs=[answer_output, urls_output])
        
        demo.load(fn=clear_in_memory_storage)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
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
from urllib.parse import urljoin, urlparse
import html2text
import re
import hashlib
import markdown2
import json
import logging
import aiofiles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

async def save_summaries_to_disk():
    async with aiofiles.open("summaries.json", "w") as f:
        await f.write(json.dumps(summary_store))

async def load_summaries_from_disk():
    global summary_store
    try:
        async with aiofiles.open("summaries.json", "r") as f:
            content = await f.read()
            summary_store = json.loads(content)
    except FileNotFoundError:
        summary_store = {}

async def process_url(session, url, depth, max_depth, max_size_bytes, semaphore, crawl_progress):
    if depth > max_depth:
        return None

    sanitized_filename = sanitize_filename(url)
    file_path = os.path.join("parsed_pages", sanitized_filename)

    try:
        crawl_progress(0, desc=f"Crawling: {url}")
        
        async with semaphore:
            # Check if the URL is already processed
            if await aiofiles.os.path.exists(file_path):
                async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
                    markdown = await file.read()
                source = "disk"
                content_size = len(markdown.encode('utf-8'))
            else:
                # Fetch and process new content
                async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        for script in soup(["script", "style"]):
                            script.decompose()
                        markdown = html_to_markdown(str(soup))
                        content_size = len(markdown.encode('utf-8'))
                        source = "network"

                        # Store the new content
                        async with aiofiles.open(file_path, "w", encoding="utf-8") as file:
                            await file.write(markdown)

        # Look for new links
        soup = BeautifulSoup(markdown, 'html.parser')
        new_urls = []
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(url, link['href'])
            if urlparse(absolute_url).netloc == urlparse(url).netloc:
                new_urls.append((absolute_url, depth + 1))

        return {
            'markdown': markdown,
            'metadata': {'sourceURL': url, 'size_bytes': content_size, 'source': source},
            'new_urls': new_urls
        }

    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        crawl_progress(0, desc=f"Error processing {url}: {str(e)}")
        return None

async def crawl_url(url, max_depth=2, max_size_mb=10, max_concurrency=5, crawl_progress=gr.Progress()):
    results = []
    total_size_bytes = 0
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    semaphore = asyncio.Semaphore(max_concurrency)

    crawl_progress(0, desc="Starting crawl...")

    async with aiohttp.ClientSession() as session:
        tasks = set([asyncio.create_task(process_url(session, url, 0, max_depth, max_size_bytes, semaphore, crawl_progress))])
        visited = set([url])

        while tasks and total_size_bytes < max_size_bytes:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = await task
                if result:
                    results.append(result)
                    total_size_bytes += result['metadata']['size_bytes']
                    
                    for new_url, new_depth in result['new_urls']:
                        if new_url not in visited and new_depth <= max_depth:
                            visited.add(new_url)
                            if total_size_bytes < max_size_bytes:
                                tasks.add(asyncio.create_task(process_url(session, new_url, new_depth, max_depth, max_size_bytes, semaphore, crawl_progress)))

            crawl_progress(0, desc=f"Processed {len(results)} pages. Total size: {total_size_bytes / (1024 * 1024):.2f} MB")

    crawl_progress(0, desc=f"Crawl complete. Total size: {total_size_bytes / (1024 * 1024):.2f} MB")
    return results

async def summarize_content_batch(contents, batch_size=5):
    summaries = []
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i+batch_size]
        chain = summarize_prompt | llm | StrOutputParser()
        batch_summaries = await asyncio.gather(*[chain.ainvoke({"content": content}) for content in batch])
        summaries.extend(batch_summaries)
    return summaries

async def crawl_and_store(url, max_size_mb=10, crawl_progress=gr.Progress()):
    global in_memory_storage, in_memory_storage_size
    try:
        await load_summaries_from_disk()
        crawl_results = await crawl_url(url, max_size_mb=max_size_mb, crawl_progress=crawl_progress)
        
        # Process results
        markdown_contents = []
        total_stored_size = 0
        
        # Prepare batch for summarization
        to_summarize = []
        for result in crawl_results:
            markdown = result['markdown']
            source_url = result['metadata']['sourceURL']
            content_size = result['metadata']['size_bytes']
            source = result['metadata']['source']
            
            sanitized_filename = sanitize_filename(source_url)
            
            if sanitized_filename not in summary_store:
                to_summarize.append((sanitized_filename, markdown))
            
            markdown_contents.append(f"Processed: {source_url} (from {source}). Size: {content_size / 1024:.2f} KB")
            total_stored_size += content_size

        # Batch summarization
        if to_summarize:
            crawl_progress(0, desc="Summarizing content...")
            summaries = await summarize_content_batch([content for _, content in to_summarize])
            for (filename, _), summary in zip(to_summarize, summaries):
                summary_store[filename] = summary.strip()

        # Save summaries to disk
        await save_summaries_to_disk()

        # Clear in-memory storage after processing
        in_memory_storage.clear()
        in_memory_storage_size = 0

        output = f"Total processed size: {total_stored_size / (1024 * 1024):.2f} MB\n" + "\n".join(markdown_contents)
        return output

    except Exception as e:
        logger.error(f"An error occurred during crawling: {str(e)}")
        return f"An error occurred during crawling: {str(e)}"

async def select_relevant_urls(query):
    try:
        summaries = "\n".join([f"{url}: {summary}" for url, summary in summary_store.items()])
        chain = select_urls_prompt | llm | StrOutputParser()
        selected_urls = await chain.ainvoke({"query": query, "summaries": summaries})
        return selected_urls.strip().split(",") if selected_urls.strip() else []
    except Exception as e:
        logger.error(f"An error occurred while selecting relevant URLs: {str(e)}")
        return []

async def answer_query(query):
    try:
        selected_urls = await select_relevant_urls(query)
        if not selected_urls:
            disclaimer = "No relevant URLs were found. "
        else:
            disclaimer = ""

        context_parts = []
        for url in selected_urls:
            sanitized_filename = sanitize_filename(url.strip())
            file_path = os.path.join("parsed_pages", sanitized_filename)
            if await aiofiles.os.path.exists(file_path):
                async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
                    context_parts.append(await file.read())

        context = "\n\n".join(context_parts)
        
        chain = answer_prompt | llm | StrOutputParser()
        answer = await chain.ainvoke({"query": query, "context": context})

        # Convert markdown to HTML
        answer_html = markdown2.markdown(answer)

        urls_output = ", ".join(selected_urls) if selected_urls else "No URLs selected."

        return disclaimer + answer_html, urls_output

    except Exception as e:
        logger.error(f"An error occurred while answering the query: {str(e)}")
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
    logger.info("Starting the Gradio interface")
    demo.launch(server_name="0.0.0.0", server_port=7860)
    logger.info("Gradio interface stopped")
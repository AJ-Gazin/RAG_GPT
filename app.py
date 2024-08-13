import gradio as gr
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from openai import OpenAI
import tiktoken
from readability import Document
import logging
from prompts import summarize_prompt, select_urls_prompt, answer_prompt
from theme import BusinessAnalyzerTheme

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MAX_PAGES = 30
MAX_TOKENS = 128000  # 128k context window
MAX_OUTPUT_TOKENS = 16000
summary_store = {}
client = None
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))

def sanitize_filename(url):
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'[\\/*?:"<>|]', '_', url)
    return url[:200]  # Limit length to 200 characters

def prioritize_pages(links):
    priority_keywords = ['about', 'services', 'products', 'contact', 'team', 'history']
    priority_pages = [link for link in links if any(keyword in link.lower() for keyword in priority_keywords)]
    return priority_pages + [link for link in links if link not in priority_pages]

def extract_relevant_text(html_content):
    # Use readability-lxml to extract main content
    doc = Document(html_content)
    return doc.summary()

def crawl_website(url, progress=gr.Progress()):
    logging.info(f"Starting crawl for: {url}")
    visited = set()
    to_visit = [url]
    pages_crawled = 0
    total_tokens = 0

    while to_visit and pages_crawled < MAX_PAGES and total_tokens < MAX_TOKENS:
        current_url = to_visit.pop(0)

        if current_url in visited:
            continue

        progress((pages_crawled / MAX_PAGES), f"Crawling: {current_url}")
        logging.info(f"Crawling: {current_url}")

        try:
            response = requests.get(current_url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                soup = BeautifulSoup(response.text, 'lxml')

                # Extract and clean text using readability
                text = extract_relevant_text(response.text)

                # Add original URL to the beginning of the text
                text = f"URL: {current_url}\n\n{text}"

                # Count tokens and check if we're within limit
                text_tokens = count_tokens(text)
                if total_tokens + text_tokens > MAX_TOKENS:
                    logging.warning("Token limit reached, stopping crawl.")
                    break

                # Store the cleaned text
                summary_store[current_url] = text
                total_tokens += text_tokens

                # Find new links
                new_links = []
                for link in soup.find_all('a', href=True):
                    new_url = urljoin(current_url, link['href'])
                    new_url = urlparse(new_url)._replace(fragment='').geturl()  # Remove fragment identifiers
                    # Ensure new_url is a full URL, within the same domain, and not already visited
                    if urlparse(new_url).netloc == urlparse(url).netloc and new_url not in visited and new_url not in to_visit:
                        new_links.append(new_url)
                        logging.info(f"Discovered new URL: {new_url}")

                # Prioritize and add new links
                prioritized_links = prioritize_pages(new_links)
                to_visit.extend(prioritized_links)

                visited.add(current_url)
                pages_crawled += 1
                logging.info(f"Pages crawled: {pages_crawled}")

        except Exception as e:
            logging.error(f"Error crawling {current_url}: {str(e)}")

    progress(1.0, f"Crawling complete. Pages crawled: {pages_crawled}")
    
    logging.info("Starting summarization process.")
    summarization_result = summarize_pages(progress)
    
    return summarization_result, f"Crawling complete. Pages crawled: {pages_crawled}."

def batch_summarize(texts, urls):
    logging.info("Batch summarization started.")
    batched_summaries = []
    current_batch = []
    current_batch_tokens = 0

    for text, url in zip(texts, urls):
        text_tokens = count_tokens(text)
        if current_batch_tokens + text_tokens > MAX_TOKENS - 1000:  # Leave room for prompt
            # Process current batch
            summaries = summarize_batch(current_batch)
            batched_summaries.extend(summaries)
            current_batch = []
            current_batch_tokens = 0

        current_batch.append((text, url))
        current_batch_tokens += text_tokens

    # Process any remaining items
    if current_batch:
        summaries = summarize_batch(current_batch)
        batched_summaries.extend(summaries)

    logging.info("Batch summarization completed.")
    return batched_summaries

def summarize_batch(batch):
    batch_text = "\n\n---\n\n".join([f"{text}" for text, url in batch])
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": summarize_prompt},
                {"role": "user", "content": batch_text}
            ],
            max_tokens=MAX_OUTPUT_TOKENS
        )
        summaries = response.choices[0].message.content.split("\n\n---\n\n")
        logging.info("Summarization successful.")
        return [summary.strip() for summary in summaries]
    except Exception as e:
        logging.error(f"Error summarizing batch: {str(e)}")
        return ["Error: An error occurred during summarization. Please check the logs."]

def summarize_pages(progress=gr.Progress()):
    global summary_store
    texts = list(summary_store.values())
    urls = list(summary_store.keys())

    summarized_store = {}
    total_batches = (len(texts) + 9) // 10  # Assuming roughly 10 pages per batch

    for i in range(0, len(texts), 10):
        batch_texts = texts[i:i+10]
        batch_urls = urls[i:i+10]
        batch_summaries = summarize_batch(list(zip(batch_texts, batch_urls)))
        
        progress((i + 10) / len(texts), f"Summarizing batch {i//10 + 1} of {total_batches}")
        logging.info(f"Summarizing batch {i//10 + 1} of {total_batches}")
        
        for url, summary in zip(batch_urls, batch_summaries):
            summarized_store[url] = summary

    summary_store = summarized_store
    progress(1.0, "Summarization complete")
    logging.info("Summarization complete.")
    return "Summarization complete."

def query_content(query):
    logging.info("Query processing started.")
    summaries = "\n".join([f"{url}: {summary}" for url, summary in summary_store.items()])

    # Select relevant URLs
    try:
        selected_urls = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": select_urls_prompt},
                {"role": "user", "content": f"Query: {query}\nSummaries:\n{summaries}"}
            ],
            max_tokens=MAX_OUTPUT_TOKENS
        ).choices[0].message.content.split(',')
        selected_urls = [url.strip() for url in selected_urls]
        logging.info("URLs selected successfully.")
    except Exception as e:
        logging.error(f"Error selecting URLs: {str(e)}")
        return "Error: An error occurred during URL selection. Please check the logs.", ""

    # Prepare context from selected URLs
    context = "\n\n".join([summary_store[url] for url in selected_urls if url in summary_store])

    # Ensure we're within token limit
    while count_tokens(context) + count_tokens(query) + count_tokens(answer_prompt) > MAX_TOKENS:
        context = "\n\n".join(context.split("\n\n")[:-1])  # Remove the last summary

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

    logging.info("Query processing completed.")
    return answer, ", ".join(selected_urls)

def set_api_key(api_key):
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

def create_interface():
    theme = BusinessAnalyzerTheme()
    
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown("# Business Website Analyzer")
        
        with gr.Tab("Setup"):
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
            api_key_output = gr.Textbox(label="Status")
            
            api_key_input.submit(
                fn=set_api_key,
                inputs=api_key_input,
                outputs=api_key_output
            )
        
        with gr.Tab("Crawl Website"):
            url_input = gr.Textbox(label="Website URL")
            crawl_button = gr.Button("Crawl and Analyze")
            crawl_output = gr.Textbox(label="Crawl Status")
            summarize_output = gr.Textbox(label="Summarization Status")
            
            crawl_button.click(
                fn=crawl_website,
                inputs=url_input,
                outputs=[summarize_output, crawl_output]
            )
        
        with gr.Tab("Query Content"):
            query_input = gr.Textbox(label="Your Query")
            query_button = gr.Button("Ask")
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")
                urls_output = gr.Textbox(label="Sources")
            query_button.click(fn=query_content, inputs=query_input, outputs=[answer_output, urls_output])
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
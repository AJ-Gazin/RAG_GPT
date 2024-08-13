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

def crawl_website(url, max_pages, progress=gr.Progress()):
    logging.info(f"Starting crawl for: {url}")
    visited = set()
    to_visit = [url]
    pages_crawled = 0
    total_tokens = 0

    while to_visit and pages_crawled < max_pages and total_tokens < MAX_TOKENS:
        current_url = to_visit.pop(0)

        if current_url in visited:
            continue

        progress((pages_crawled / max_pages), f"Crawling: {current_url}")
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
    with gr.Blocks(css=".center { text-align: center; width: 100%; } .radio-button { flex: 1; text-align: center; }", theme=BusinessAnalyzerTheme()) as demo:
        gr.Markdown("# Business Website Analyzer", elem_classes="center")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Setup", elem_classes="center")
                with gr.Row():
                    api_key_input = gr.Textbox(label="OpenAI API Key", type="password", show_label=False, placeholder="Enter OpenAI API Key", elem_classes="center")
                    api_status = gr.Markdown(elem_id="api-status")
                api_key_button = gr.Button("Set API Key", variant="primary")

                gr.Markdown("## Website Crawling", elem_classes="center")
                url_input = gr.Textbox(label="Website URL", show_label=False, placeholder="Enter Website URL", elem_classes="center")
                crawl_depth = gr.Radio(
                    ["Shallow (15 pages)", "Robust (30 pages)", "Comprehensive (60 pages)"],
                    label="Crawl Depth",
                    value="Robust (30 pages)",
                    show_label=False,
                    elem_classes="center radio-button"
                )
                crawl_button = gr.Button("Crawl and Analyze", variant="primary")
                
                gr.Markdown("## Content Query", elem_classes="center")
                query_input = gr.Textbox(label="Enter Your Query", show_label=False, placeholder="Type Your Query", elem_classes="center")
                query_button = gr.Button("Ask", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("## Crawl Results", elem_classes="center")
                crawl_output = gr.Textbox(label="Crawl Status", show_label=False, placeholder="Crawl Status")
                summarize_output = gr.Textbox(label="Summarization Status", show_label=False, placeholder="Summarization Status")
                
                gr.Markdown("## Query Results", elem_classes="center")
                answer_output = gr.Markdown()
                urls_output = gr.Textbox(label="Sources", show_label=False, placeholder="Sources")

        # Event handlers
        api_key_button.click(fn=set_api_key, inputs=api_key_input, outputs=api_status)
        crawl_button.click(fn=lambda url, depth: crawl_website(url, {"Shallow (15 pages)": 15, "Robust (30 pages)": 30, "Comprehensive (60 pages)": 60}[depth]), inputs=[url_input, crawl_depth], outputs=[crawl_output, summarize_output])
        query_button.click(fn=query_content, inputs=query_input, outputs=[answer_output, urls_output])

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
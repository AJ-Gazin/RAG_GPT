import gradio as gr
from firecrawl import FirecrawlApp
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize Firecrawl
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=firecrawl_api_key)

def crawl_and_display(url):
    try:
        # Start a crawl with depth 1
        crawl_results = app.crawl_url(url, {
            'crawlerOptions': {
                'maxDepth': 1,
                'onlyMainContent': True
            },
            'wait_until_done': True
        })

        # Ensure crawl_results is a list
        if not isinstance(crawl_results, list):
            return "Unexpected response from crawl job. Please check the URL and try again."

        # Gather markdown content from results
        markdown_contents = []
        for result in crawl_results:
            markdown = result.get('markdown', '')
            markdown_contents.append(markdown)

        # Join and return markdown content for display
        return "\n\n---\n\n".join(markdown_contents)

    except Exception as e:
        return f"An error occurred during crawling: {str(e)}"

# Gradio interface
demo = gr.Interface(
    fn=crawl_and_display,
    inputs=gr.Textbox(label="Website URL"),
    outputs=gr.Markdown(label="Crawled Markdown Content"),
    title="Firecrawl Test",
    description="Enter a website URL to crawl with a depth of 1 and display the markdown content.",
)

demo.launch()

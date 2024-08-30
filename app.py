import gradio as gr
from logic import crawl_website, analyze_website, query_content
from theme import BusinessAnalyzerTheme
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %Y-%m-%d %H:%M:%S')

def create_interface():
    logging.info("Creating Gradio interface")
    css = """
    h1 {
        text-align: center;
        display:block;
    }
    """
    with gr.Blocks(css=css, theme=BusinessAnalyzerTheme()) as demo:
        gr.Markdown("# Business Website Analyzer", elem_classes="center")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1) Crawl and Analyze Website", elem_classes="center")
                url_input = gr.Textbox(label="Website URL", placeholder="Enter Website URL")
                crawl_depth = gr.Radio(
                    ["Shallow (5 pages)", "Robust (30 pages)", "Comprehensive (60 pages)"],
                    label="Crawl Depth",
                    value="Robust (30 pages)"
                )
                crawl_button = gr.Button("Crawl and Analyze Website", variant="primary")
                crawl_status = gr.HTML("<center>Ready to Crawl and Analyze!</center>")
                analysis_status = gr.HTML("<center></center>")
                
                gr.Markdown("## 2) Ask Questions", elem_classes="center")
                query_input = gr.Textbox(label="Enter Your Query", placeholder="Type Your Query")
                query_button = gr.Button("Ask", variant="primary")

                gr.Markdown("## Knowledge Graph Visualization", elem_classes="center")
                graph_html = gr.HTML(visible=False)

            with gr.Column(scale=1):
                gr.Markdown("## Query Results", elem_classes="center")
                answer_output = gr.Markdown()
                urls_output = gr.Textbox(label="Sources")

        def crawl_wrapper(url, depth, progress=gr.Progress()):
            logging.info(f"Starting crawl for URL: {url} with depth: {depth}")
            max_pages = {"Shallow (5 pages)": 5, "Robust (30 pages)": 30, "Comprehensive (60 pages)": 60}[depth]
            for pages_crawled, total_pages, status in crawl_website(url, max_pages):
                progress(pages_crawled / total_pages, status)
            return f"<center>Crawling complete. Pages crawled: {pages_crawled}</center>"

        def start_analysis():
            logging.info("Starting analysis")
            return "<center>Analysis in progress...</center>"

        def analyze_wrapper():
            logging.info("Running analysis")
            analysis_message, graph_html_content = analyze_website()
            return f"<center>{analysis_message}</center>", graph_html_content, gr.update(visible=True)

        crawl_button.click(
            fn=crawl_wrapper,
            inputs=[url_input, crawl_depth],
            outputs=crawl_status
        ).success(
            fn=start_analysis,
            outputs=analysis_status
        ).then(
            fn=analyze_wrapper,
            outputs=[analysis_status, graph_html, graph_html]
        )
        
        query_button.click(fn=query_content, inputs=query_input, outputs=[answer_output, urls_output])

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
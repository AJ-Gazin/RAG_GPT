import gradio as gr
from logic import crawl_website, analyze_website, query_content, load_example_graph, query_example_graph, generate_graph_visualization
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %Y-%m-%d %H:%M:%S')

# Global variable to store the current graph visualization
current_graph_html = ""

def create_interface():
    logging.info("Creating Gradio interface")
    
    def crawl_wrapper(url, depth):
        global current_graph_html
        logging.info(f"Starting crawl for URL: {url} with depth: {depth}")
        max_pages = {"Shallow (5 pages)": 5, "Robust (30 pages)": 30, "Comprehensive (60 pages)": 60}[depth]
        for pages_crawled, total_pages, status in crawl_website(url, max_pages):
            yield f"<center>{status}</center>"
        yield f"<center>Crawling complete. Pages crawled: {pages_crawled}</center>"

    def start_analysis():
        logging.info("Starting analysis")
        return "<center>Analysis in progress...</center>"

    def analyze_wrapper():
        global current_graph_html
        logging.info("Running analysis")
        analysis_message, graph_html_content = analyze_website()
        current_graph_html = graph_html_content
        return f"<center>{analysis_message}</center>", graph_html_content, gr(visible=True)

    def load_example_wrapper():
        global current_graph_html
        logging.info("Loading NeuronsLab example")
        graph_store = load_example_graph()
        current_graph_html = generate_graph_visualization(graph_store)
        return "<center>NeuronsLab example loaded successfully</center>", current_graph_html, gr(visible=True)

    def query_wrapper(query, website_choice):
        logging.info(f"Processing query: {query}")
        if website_choice == "Custom Website":
            response, urls = query_content(query)
        else:
            response, urls = query_example_graph(query)
        return response, urls

    with gr.Blocks() as demo:
        gr.Markdown("# Business Website Analyzer", elem_id="title")
        
        with gr.Row():
            with gr.Column(scale=1):
                website_choice = gr.Radio(["Custom Website", "NeuronsLab Example"], label="Website Choice", value="Custom Website")
                
                with gr.Group() as custom_website_group:
                    url_input = gr.Textbox(label="Website URL", placeholder="Enter Website URL")
                    crawl_depth = gr.Radio(
                        ["Shallow (5 pages)", "Robust (30 pages)", "Comprehensive (60 pages)"],
                        label="Crawl Depth",
                        value="Robust (30 pages)"
                    )
                    crawl_button = gr.Button("Crawl and Analyze Website")
                
                with gr.Group(visible=False) as example_website_group:
                    load_example_button = gr.Button("Load NeuronsLab Example")
                
                crawl_status = gr.HTML("<center>Ready to Crawl and Analyze!</center>")
                analysis_status = gr.HTML()
                
                query_input = gr.Textbox(label="Enter Your Query", placeholder="Type Your Query")
                query_button = gr.Button("Ask")

                graph_html = gr.HTML(visible=False, label="Knowledge Graph Visualization")

            with gr.Column(scale=1):
                answer_output = gr.Markdown(label="Query Results")
                urls_output = gr.Textbox(label="Sources")

        def toggle_website_choice(choice):
            if choice == "Custom Website":
                return gr.Group(visible=True), gr.Group(visible=False)
            else:
                return gr.Group(visible=False), gr.Group(visible=True)

        website_choice.change(toggle_website_choice, inputs=[website_choice], outputs=[custom_website_group, example_website_group])

        crawl_button.click(
            fn=crawl_wrapper,
            inputs=[url_input, crawl_depth],
            outputs=crawl_status
        ).then(
            fn=start_analysis,
            outputs=analysis_status
        ).then(
            fn=analyze_wrapper,
            outputs=[analysis_status, graph_html, graph_html]
        )
        
        load_example_button.click(
            fn=load_example_wrapper,
            outputs=[analysis_status, graph_html, graph_html]
        )
        
        query_button.click(
            fn=query_wrapper,
            inputs=[query_input, website_choice],
            outputs=[answer_output, urls_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(allowed_paths=["."])  # Allow serving files from the current directory
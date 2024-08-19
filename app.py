import gradio as gr
from theme import BusinessAnalyzerTheme
from logic import set_api_key, crawl_website, query_content

def create_interface():
    """Create the Gradio interface for the Business Website Analyzer."""
    with gr.Blocks(css=".center { text-align: center; width: 100%; } .radio-button { flex: 1; text-align: center; }", theme=BusinessAnalyzerTheme()) as demo:
        gr.Markdown("# Business Website Analyzer", elem_classes="center")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Step 1: API Key Input
                gr.Markdown("## 1) Enter your OpenAI API key", elem_classes="center")
                with gr.Row():
                    api_key_input = gr.Textbox(label="OpenAI API Key", type="password", show_label=False, placeholder="Enter OpenAI API Key", elem_classes="center")
                    api_status = gr.Markdown(elem_id="api-status")
                api_key_button = gr.Button("Set API Key", variant="primary")

                # Step 2: Website Selection and Crawling
                gr.Markdown("## 2) Select a Website!", elem_classes="center")
                url_input = gr.Textbox(label="Website URL", show_label=False, placeholder="Enter Website URL", elem_classes="center")
                crawl_depth = gr.Radio(
                    ["Shallow (15 pages)", "Robust (30 pages)", "Comprehensive (60 pages)"],
                    label="Crawl Depth",
                    value="Robust (30 pages)",
                    show_label=False,
                    elem_classes="center radio-button"
                )
                crawl_button = gr.Button("Crawl and Analyze", variant="primary")
                
                # Step 3: Query Input
                gr.Markdown("## 3) Ask the Website any Question", elem_classes="center")
                query_input = gr.Textbox(label="Enter Your Query", show_label=False, placeholder="Type Your Query", elem_classes="center")
                query_button = gr.Button("Ask", variant="primary")

            with gr.Column(scale=1):
                # Crawl Results Display
                gr.Markdown("## Crawl Results", elem_classes="center")
                crawl_output = gr.Textbox(label="Crawl Status", show_label=False, placeholder="Crawl Status")
                summarize_output = gr.Textbox(label="Summarization Status", show_label=False, placeholder="Summarization Status")
                
                # Query Results Display
                gr.Markdown("## Query Results", elem_classes="center")
                answer_output = gr.Markdown()
                urls_output = gr.Textbox(label="Sources", show_label=False, placeholder="Sources")

        # Knowledge Graph Display
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Knowledge Graph", elem_classes="center")
                graph_output = gr.Plot()
        
        # Event handlers
        api_key_button.click(
            fn=set_api_key,
            inputs=api_key_input,
            outputs=api_status
        )
        
        crawl_button.click(
            fn=lambda url, depth: crawl_website(url, {"Shallow (15 pages)": 15, "Robust (30 pages)": 30, "Comprehensive (60 pages)": 60}[depth]), 
            inputs=[url_input, crawl_depth], 
            outputs=[crawl_output, summarize_output, graph_output]
        )
        
        query_button.click(
            fn=query_content,
            inputs=query_input,
            outputs=[answer_output, urls_output]
        )

    return demo

if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_interface()
    demo.launch()
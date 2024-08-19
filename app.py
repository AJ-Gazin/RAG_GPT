import gradio as gr
from theme import BusinessAnalyzerTheme
from logic import set_api_key, crawl_website, query_content

def create_interface():
    """Create the Gradio interface for the Business Website Analyzer."""
    with gr.Blocks(theme=BusinessAnalyzerTheme()) as demo:
        gr.Markdown("# Business Website Analyzer")
        
        with gr.Row():
            with gr.Column():
                # Step 1: API Key Input
                gr.Markdown("## 1) Enter your OpenAI API key")
                api_key_input = gr.Textbox(label="OpenAI API Key", type="password", placeholder="Enter OpenAI API Key")
                api_status = gr.Markdown()
                api_key_button = gr.Button("Set API Key")

                # Step 2: Website Selection and Crawling
                gr.Markdown("## 2) Select a Website")
                url_input = gr.Textbox(label="Website URL", placeholder="Enter Website URL")
                crawl_depth = gr.Radio(
                    ["Quick (15 pages)", "Robust (30 pages)", "Comprehensive (60 pages)"],
                    label="Crawl Depth",
                    value="Robust (30 pages)"
                )
                crawl_button = gr.Button("Crawl and Analyze")
                
                # Step 3: Query Input
                gr.Markdown("## 3) Ask the Website any Question")
                query_input = gr.Textbox(label="Enter Your Query", placeholder="Type Your Query")
                query_button = gr.Button("Ask")

            with gr.Column():
                # Results Display
                crawl_output = gr.Textbox(label="Crawl Status")
                summarize_output = gr.Textbox(label="Summarization Status")
                answer_output = gr.Markdown(label="Answer")
                urls_output = gr.Textbox(label="Sources")

        # Knowledge Graph Display
        graph_output = gr.Plot(label="Knowledge Graph")
        
        # Event handlers
        api_key_button.click(
            fn=set_api_key,
            inputs=api_key_input,
            outputs=api_status
        )
        
        crawl_button.click(
            fn=lambda url, depth: crawl_website(url, {"Quick (15 pages)": 15, "Robust (30 pages)": 30, "Comprehensive (60 pages)": 60}[depth]), 
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
    demo = create_interface()
    demo.launch()
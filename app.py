import gradio as gr
from logic import process_inputs, set_api_keys

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Webpage Parser and RAG-based Query Answering with Link Crawling")

        with gr.Row():
            firecrawl_key_input = gr.Textbox(label="Firecrawl API Key", type="password")
            openai_key_input = gr.Textbox(label="OpenAI API Key", type="password")
            set_keys_btn = gr.Button("Set API Keys")
        
        status_output = gr.Textbox(label="Status", interactive=False)
        set_keys_btn.click(set_api_keys, [firecrawl_key_input, openai_key_input], status_output)
        
        with gr.Row():
            url_input = gr.Textbox(label="URL")
            query_input = gr.Textbox(label="Query")
        
        answer_output = gr.Textbox(label="Answer")
        submit_btn = gr.Button("Submit")
        submit_btn.click(process_inputs, [url_input, query_input], answer_output)

    demo.launch(share=False)

if __name__ == "__main__":
    build_interface()

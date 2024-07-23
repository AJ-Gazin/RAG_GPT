import gradio as gr
from logic import parse_webpage, get_relevant_hyperlinks, follow_and_parse_hyperlinks, setup_rag, answer_query

# Global variables to cache API keys
firecrawl_api_key = None
openai_api_key = None

def set_api_keys(firecrawl_key, openai_key):
    global firecrawl_api_key, openai_api_key
    firecrawl_api_key = firecrawl_key
    openai_api_key = openai_key
    return "API keys have been set."

def process_inputs(url, query):
    global firecrawl_api_key, openai_api_key
    
    if firecrawl_api_key is None or openai_api_key is None:
        return "Please set the API keys first."
    
    original_file_path, hyperlinks = parse_webpage(url, firecrawl_api_key)
    relevant_hyperlinks = get_relevant_hyperlinks(openai_api_key, query, hyperlinks)
    additional_documents = follow_and_parse_hyperlinks(relevant_hyperlinks, firecrawl_api_key)
    
    all_documents = [original_file_path] + additional_documents
    documents = setup_rag(openai_api_key, all_documents)
    answer = answer_query(query, documents, openai_api_key)
    return answer

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

demo.launch(share=True)

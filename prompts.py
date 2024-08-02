# prompts.py

from langchain_core.prompts import PromptTemplate

# Prompt to summarize content
summarize_prompt = PromptTemplate.from_template(
    "Summarize the following content in 1-2 sentences:\n{content}"
)

# Prompt to select relevant URLs
select_urls_prompt = PromptTemplate.from_template(
    "Given the query: {query}\nAnd the following summaries:\n{summaries}\n"
    "Select the most relevant URLs for answering the query. Return only the URLs, separated by commas, with a maximum of 2."
)

# Prompt to provide a comprehensive answer with detailed instructions
answer_prompt = PromptTemplate.from_template(
    "Given the query: {query}\nAnd the following context:\n{context}\n"
    "Instructions:\n"
    "- Provide a detailed and thorough answer to the user's query.\n"
    "- Use information from the provided context to support your answer.\n"
    "- Include relevant examples or explanations to enhance the clarity of your response.\n"
    "- Ensure your answer is well-structured and easy to understand.\n"
    "- If there are multiple points of view or interpretations, present them clearly.\n"
    "Provide a comprehensive answer to the query."
)

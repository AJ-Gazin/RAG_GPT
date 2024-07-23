def get_relevant_hyperlinks_prompt(query, hyperlinks):
    prompt = f"User query: {query}\n\nHyperlinks:\n" + "\n".join(hyperlinks) + "\n\nBased on the user query, which of these hyperlinks are relevant? List the relevant hyperlinks."
    return prompt

def answer_query_prompt(query, combined_document):
    prompt = f"""
You are an intelligent assistant tasked with providing detailed and comprehensive answers to user queries based on the provided documents. Below is a user's query followed by relevant documents that you can use to generate your answer.

User Query:
{query}

Relevant Documents:
{combined_document}

Instructions:
- Provide a detailed and thorough answer to the user's query.
- Use information from the provided documents to support your answer.
- Include relevant examples or explanations to enhance the clarity of your response.
- Ensure your answer is well-structured and easy to understand.
- If there are multiple points of view or interpretations, present them clearly.

Answer:
"""
    return prompt

def score_hyperlinks_prompt(query, hyperlinks_with_context):
    prompt = f"User query: {query}\n\nHyperlinks and Contexts:\n"
    for link, context in hyperlinks_with_context:
        prompt += f"{link}\ncontext: {context}\n"
    prompt += "\nPlease score each hyperlink based on its relevancy to the user's query. Format: url score (e.g., http://example.com 0.85)"
    return prompt
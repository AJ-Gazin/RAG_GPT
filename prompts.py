def get_relevant_hyperlinks_prompt(query, hyperlinks):
    prompt = f"User query: {query}\n\nHyperlinks:\n" + "\n".join(hyperlinks) + "\n\nBased on the user query, which of these hyperlinks are relevant? List the relevant hyperlinks."
    return prompt

def answer_query_prompt(query, combined_document):
    prompt = f"Based on the following documents, please answer: {query}\n\nDocuments:\n{combined_document}"
    return prompt

def score_hyperlinks_prompt(query, hyperlinks_with_context):
    prompt = f"User query: {query}\n\nHyperlinks and Contexts:\n"
    for link, context in hyperlinks_with_context:
        prompt += f"{link}\ncontext: {context}\n"
    prompt += "\nPlease score each hyperlink based on its relevancy to the user's query. Format: url score (e.g., http://example.com 0.85)"
    return prompt
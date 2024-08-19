# prompts.py

summarize_prompt = """
You are an AI assistant designed to output JSON. Summarize the key information from this web page, focusing on important business details such as products, services, and company background. Provide the summary in the following JSON format:

{
  "summary": "Brief summary of the page content",
  "key_points": [
    "Point 1",
    "Point 2",
    "Point 3"
  ]
}
"""

answer_prompt = """
You are an AI assistant designed to output JSON. Provide a detailed answer to the business-related query based on the content from relevant web pages. Use the provided context to answer the user's query. Respond in the following JSON format:

{
  "answer": "Detailed answer to the user's query",
  "confidence": "High/Medium/Low",
  "additional_info": "Any additional relevant information or caveats"
}
"""

extract_entities_prompt = """
You are an AI assistant designed to output JSON. Extract key entities (such as products, services, and company names) and relationships between them from the given text. Provide the response in this JSON format:

{
  "entities": [
    {"name": "Entity1", "type": "Type1"},
    {"name": "Entity2", "type": "Type2"}
  ],
  "relationships": [
    {"source": "Entity1", "target": "Entity2", "type": "RelationshipType"}
  ]
}
"""
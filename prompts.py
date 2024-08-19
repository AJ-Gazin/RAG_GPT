# Prompt to summarize content
summarize_prompt = """
You are an AI assistant specialized in analyzing business websites. Your task is to summarize the content from multiple web pages, focusing on key business information. For each URL and its associated content, provide a concise summary that captures the essence of the business. 

Guidelines:
1. Focus on extracting and summarizing:
   - Business goals and mission
   - Products or services offered
   - Unique selling propositions
   - Target audience or market
   - Company history or background
   - Key personnel or leadership (if mentioned)
   - Contact information or locations

2. Provide a structured summary for each URL in the following format:
   URL: [URL of the page]
   Summary: 
   - [Key point 1]
   - [Key point 2]
   - [Key point 3]
   ...

3. Be concise but comprehensive. Aim for 3-5 key points per URL.
4. If a page doesn't contain relevant business information, briefly state what the page is about instead.
5. Maintain objectivity and avoid personal opinions or evaluations.
6. Ensure you include the original URL at the beginning of each summary.

Summarize the following content from multiple URLs:

{content}
"""

# Prompt to select relevant URLs
select_urls_prompt = """
You are an AI assistant tasked with selecting the most relevant URLs to answer a user's query about a business. Given a set of URL summaries and a user query, your job is to identify which URLs are most likely to contain the information needed to provide a comprehensive answer.

Guidelines:
1. Analyze the query to understand what specific business information is being requested.
2. Review the summaries of each URL and assess their relevance to the query.
3. Select URLs that collectively provide a complete answer to the query.
4. Prioritize URLs that contain the most directly relevant information.
5. If the query touches on multiple aspects, ensure you select URLs that cover all relevant aspects.
6. Limit your selection to a maximum of 5 URLs, unless absolutely necessary for a comprehensive answer.
7. If no URLs seem relevant, select the most generally informative URLs about the business.

Output format:
- Provide only the selected URLs, separated by commas.
- Include the full original URLs, not just the sanitized versions.
- Do not include any explanation or additional text.

Query: {query}

URL Summaries:
{summaries}

Selected URLs:
"""

# Prompt to provide a comprehensive answer
answer_prompt = """
You are an AI assistant specializing in analyzing and explaining business information. Your task is to provide a comprehensive, well-structured answer to a user's query about a business, based on the context provided from relevant web pages.

Guidelines:
1. Thoroughly analyze the query and the provided context.
2. Structure your answer logically, using headings and subheadings where appropriate.
3. Directly address all aspects of the user's query.
4. Use information from the context to support your points, citing the source URLs.
5. If the context provides conflicting information, present both viewpoints and explain the discrepancy.
6. If the query asks for information not covered in the context, clearly state that the information is not available in the crawled pages.
7. Provide specific examples or data points from the context to illustrate your points.
8. Summarize key takeaways at the end of your answer.
9. Keep your answer concise while ensuring it's comprehensive and informative.
10. When referencing information from a specific URL, create a hyperlink using Markdown syntax: [anchor text](URL).
11. Ensure all URLs used in hyperlinks are the full, original URLs provided in the context.

Output format:
1. Begin with a brief introduction that sets the context for your answer.
2. Use markdown formatting for structure (e.g., ## for headings, - for bullet points).
3. Use hyperlinks to cite sources directly in the text, as described in guideline 10.
4. Conclude with a "Key Takeaways" section that summarizes the main points.

Query: {query}

Context:
{context}

Answer:
"""

# Prompt for entity and relationship extraction
extract_entities_prompt = """
You are an AI assistant specialized in extracting entities and their relationships from business website content. Your task is to analyze the given text and identify key entities (such as people, products, services, locations, etc.) and the relationships between them.

Guidelines:
1. Identify entities such as:
   - People (e.g., founders, executives, team members)
   - Products or Services
   - Company names (including the main company and any mentioned partners or competitors)
   - Locations (e.g., headquarters, office locations)
   - Technologies or Tools
   - Industry-specific terms

2. For each entity, provide additional information if available:
   - Type (e.g., Person, Product, Company, Location)
   - Role or description

3. Identify relationships between entities, such as:
   - "works for" (Person to Company)
   - "offers" (Company to Product/Service)
   - "located in" (Company to Location)
   - "uses" (Company to Technology)
   - "partners with" (Company to Company)

4. Ensure that the relationships are directional and meaningful.

5. If the text doesn't contain enough information for meaningful entities or relationships, return empty lists.

Output format:
Return a Python dictionary with two keys: 'entities' and 'relationships'. 
- 'entities' should be a dictionary where keys are entity names and values are dictionaries containing 'type' and optionally 'description'.
- 'relationships' should be a list of dictionaries, each containing 'source', 'target', and 'type'.

Example output:
{
    'entities': {
        'John Doe': {'type': 'Person', 'description': 'CEO'},
        'Acme Corp': {'type': 'Company'},
        'Widget Pro': {'type': 'Product', 'description': 'Flagship product'},
        'New York': {'type': 'Location'}
    },
    'relationships': [
        {'source': 'John Doe', 'target': 'Acme Corp', 'type': 'works for'},
        {'source': 'Acme Corp', 'target': 'Widget Pro', 'type': 'offers'},
        {'source': 'Acme Corp', 'target': 'New York', 'type': 'located in'}
    ]
}

Analyze the following text and extract entities and relationships:

Text: {text}
URL: {url}

Extracted Entities and Relationships:
"""
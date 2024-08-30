# prompts.py

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

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document parsed from the html of a business's website, identify all entities and their entity types from the text and all relationships among the identified entities. Particularly, focus on information a user may want to ask about a business, such as employees, services, ideals, etc.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$<entity_name>$$$$<entity_type>$$$$<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_description>)

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:
"""

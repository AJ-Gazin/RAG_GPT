import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Example text to test triplet extraction
sample_text = """
. 
"""
with open("C:\\Users\\aj\\Code\\RAG_GPT\\parsed_pages\\en.wikipedia.org_wiki_Genome.md", "r", encoding="utf-8") as file:
    sample_text = file.read()

def test_kg_triplet_extract_fn(text):
    prompt = f"""
    Extract key information from the following text as a list of triplets in the format (entity1, relation, entity2).
    Focus on main topics, key facts, and relationships between concepts.
    Text content:
    {text}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts key information as triplets."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1600  # Adjust token limit based on expected output size
    )
    
    triplets = []
    # Extract triplets from the response
    response_text = response.choices[0].message.content
    for line in response_text.split('\n'):
        line = line.strip()
        # Check for valid triplet format
        if line.startswith('(') and line.endswith(')'):
            try:
                triplet = eval(line)
                triplets.append(triplet)
            except Exception as e:
                print(f"Failed to parse line: {line} - Error: {str(e)}")
        else:
            # Try to clean up and evaluate potential triplets that aren't formatted as expected
            try:
                # Attempt to reformat potential triplets
                triplet = tuple(line.strip('()').split(', '))
                if len(triplet) == 3:
                    triplets.append(triplet)
            except Exception as e:
                print(f"Failed to reformat line: {line} - Error: {str(e)}")
    
    return triplets

# Run the test
triplets = test_kg_triplet_extract_fn(sample_text)
print("Extracted Triplets:")
for triplet in triplets:
    print(triplet)

# You can add more test cases by modifying sample_text or use a list of sample texts.

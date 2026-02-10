from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

# Local LLM (Ollama)
llm = ChatOllama(
    model="llama3",
    temperature=0
)

# --------------------------------------------------
# Tesla text to chunk
# --------------------------------------------------

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs.
"""

# --------------------------------------------------
# Prompt-based (LLM-assisted) chunking
# --------------------------------------------------

prompt = f"""
You are a text chunking expert.

Split the following text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put <<<SPLIT>>> between chunks

Text:
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split.
"""

print("🤖 Asking local LLM (Ollama) to chunk the text...")

response = llm.invoke(prompt)
marked_text = response.content

# --------------------------------------------------
# Post-processing
# --------------------------------------------------

chunks = marked_text.split("<<<SPLIT>>>")

clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        clean_chunks.append(cleaned)

# --------------------------------------------------
# Output
# --------------------------------------------------

print("\n🎯 LLM-ASSISTED CHUNKING RESULTS")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(chunk)

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly.
"""

# --------------------------------------------------
# Example 1: CharacterTextSplitter (naive, NOT recommended)
# --------------------------------------------------
# Uncomment only if you want to see bad behaviour

# splitter1 = CharacterTextSplitter(
#     separator=" ",
#     chunk_size=100,
#     chunk_overlap=0
# )
#
# chunks1 = splitter1.split_text(tesla_text)
# for i, chunk in enumerate(chunks1, 1):
#     print(f"Chunk {i}: ({len(chunk)} chars)")
#     print(chunk)
#     print()

# --------------------------------------------------
# Example 2: RecursiveCharacterTextSplitter (CORRECT)
# --------------------------------------------------

print("\n" + "=" * 60)
print("RECURSIVE CHARACTER TEXT SPLITTER (HF READY)")
print("=" * 60)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = recursive_splitter.split_text(tesla_text)

for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(chunk)

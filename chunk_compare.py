"""
Chunking Strategy Comparison
- Strategy 1: fixed-size chunks (300 characters, 50 character overlap)
- Strategy 2: paragraph-based chunks (split on double newlines)

Both sets are embedded, searched with the same 3 queries, and the
top 2 results from each strategy are printed with similarity scores.
"""

from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------
# Step 1: Load the document
# ---------------------------------------------------------------
with open("document.txt", "r", encoding="utf-8") as f:
    text = f.read()


# ---------------------------------------------------------------
# Step 2: Chunking strategies
# ---------------------------------------------------------------
def fixed_size_chunks(text, size=300, overlap=50):
    """Cut the text every `size` characters, stepping back `overlap`
    characters each time so neighboring chunks share some text."""
    chunks = []
    step = size - overlap  # 300 - 50 = move forward 250 chars each time
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):  # last chunk reached the end
            break
    return chunks


def paragraph_chunks(text):
    """Split wherever there is a blank line (double newline)."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


fixed_chunks = fixed_size_chunks(text)
para_chunks = paragraph_chunks(text)

print(f"Fixed-size strategy:      {len(fixed_chunks)} chunks")
print(f"Paragraph-based strategy: {len(para_chunks)} chunks")


# ---------------------------------------------------------------
# Step 3: Embed all chunks from both strategies
# ---------------------------------------------------------------
print("Model is loading...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded. Embedding chunks...")

fixed_embeddings = model.encode(fixed_chunks, convert_to_tensor=True)
para_embeddings = model.encode(para_chunks, convert_to_tensor=True)


# ---------------------------------------------------------------
# Step 4: Search function
# ---------------------------------------------------------------
def search(query, chunks, embeddings, top_k=2):
    """Return the top_k (chunk, score) pairs for a query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top = scores.topk(k=top_k)
    return [(chunks[int(i)], float(s)) for s, i in zip(top.values, top.indices)]


def show_results(label, results):
    print(f"\n  {label}")
    for rank, (chunk, score) in enumerate(results, start=1):
        preview = chunk.replace("\n", " ")
        if len(preview) > 150:
            preview = preview[:150] + "..."
        print(f"    #{rank}  score={score:.3f}  |  {preview}")


# ---------------------------------------------------------------
# Step 5: Run 3 queries against both strategies
# ---------------------------------------------------------------
queries = [
    "How do I validate the data sent to my API?",
    "How can I make a button do something when clicked?",
    "How does a web page load data from a server?",
]

for query in queries:
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    show_results(
        "Fixed-size (300 / 50 overlap):", search(query, fixed_chunks, fixed_embeddings)
    )
    show_results("Paragraph-based:", search(query, para_chunks, para_embeddings))


# ---------------------------------------------------------------
# Step 6: Comparison
# ---------------------------------------------------------------
COMPARISON = """
COMPARISON
Fixed-size chunking produced the higher top score on all 3 queries
(0.474 vs 0.465, 0.433 vs 0.317, 0.599 vs 0.571), but paragraph-based
chunking returned the correct topic as its #1 result on all 3 queries,
while fixed-size missed on the validation query. Its top chunk there was
about fetching data, and the actual validation content only came in at #2.

Fixed-size chunks score higher because they are short (18 chunks vs 7),
so each one holds only a few sentences and its embedding is very focused.
This helped most on the button-click query, where a small chunk about
event listeners matched better than the full JavaScript paragraph, which
mixes several ideas. However, fixed-size chunks often start or end
mid-sentence and can blend two topics, which caused the wrong match on
the validation query.

Paragraph chunks are complete, include their headings, and keep one
topic together, so they are more reliable and give better context if the
result is passed to an AI to answer the question. Overall, paragraph-based
chunking was more accurate, but similarity scores alone can be misleading
when comparing chunks of different sizes.
"""
print("\n" + COMPARISON)

from search import search

queries = [
    "Where is ingestion implemented?",
    "How does search work?",
]

for q in queries:
    results = search(q)
    print(f"\nQuery: {q}")
    for r in results:
        print(f"{r['filepath']}:{r['start_line']}")
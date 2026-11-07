from app.embedding import embeddings

document = """
Neuro Mates Parkinson's module uses
ESP32 and MPU6050 sensors to monitor
hand movement and unusual movement patterns.
"""
print("Generating document embedding...")

document_vector = embeddings.embed_documents([document])[0]

print("Document embedding successful!")
print("Vector dimensions:", len(document_vector))
print("First 10 values:", document_vector[:10])

query = "Which sensor is used for Parkinson's monitoring?"

print("\nGenerating query embedding...")

query_vector = embeddings.embed_query(query)

print("Query embedding successful!")
print("Vector dimensions:", len(query_vector))

print("\nEmbedding validation")

print("Document vector dimension:", len(document_vector))
print("Query vector dimension:", len(query_vector))

print("Configured model:", embeddings.model)
print(
    "Configured dimensions:",
    embeddings.output_dimensionality
)

assert len(document_vector) == len(query_vector)

print("\nEmbedding validation passed!")



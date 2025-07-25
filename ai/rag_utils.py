import os
from ai.agent import AttierlyAIAgent

def ingest_wardrobe_and_dataset(wardrobe_folder, dataset_folder):
    """
    Ingest wardrobe image metadata and fashion dataset into the vector store for RAG.
    """
    docs = []
    # Ingest wardrobe items
    for filename in os.listdir(wardrobe_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Use filename as metadata, could be improved with classification
            docs.append(f"Wardrobe item: {filename}")
    # Ingest dataset items (CSV or images)
    styles_csv = os.path.join(dataset_folder, 'styles.csv')
    if os.path.exists(styles_csv):
        import pandas as pd
        df = pd.read_csv(styles_csv, on_bad_lines='skip')
        for _, row in df.iterrows():
            desc = f"Dataset item: {row.get('baseColour', '')} {row.get('articleType', '')} (id={row.get('id', '')})"
            docs.append(desc)
    # Ingest into vector store
    agent = AttierlyAIAgent()
    agent.ingest_documents(docs)
    print(f"Ingested {len(docs)} documents into vector store.") 
import os
import re
from pathlib import Path
from typing import List, Dict

import pytesseract
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from opensearchpy import OpenSearch

load_dotenv()

INDEX_NAME = "rockwell_rag"
DATA_DIR = Path("data/raw")

openai_client = OpenAI(api_key=os.getenv("ORGA_OPENAI_API_KEY"))

os_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False,
    verify_certs=False
)


def run_ocr_tif(file_path: Path) -> str:
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang="eng")
    return text


def clean_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_metadata(text: str) -> Dict[str, str]:
    part_number = ""
    invoice_number = ""
    customer_po = ""

    part_match = re.search(r"\b\d{3}-\d{4}-\d{3}\b", text)
    if part_match:
        part_number = part_match.group(0)

    invoice_match = re.search(
        r"Invoice\s*(No\.?|Number)?\s*[:#]?\s*([A-Z0-9\-]+)",
        text,
        re.I
    )
    if invoice_match:
        invoice_number = invoice_match.group(2)

    po_match = re.search(
        r"(Customer\s*P\.?O\.?|PO)\s*[:#]?\s*([A-Z0-9\-]+)",
        text,
        re.I
    )
    if po_match:
        customer_po = po_match.group(2)

    return {
        "part_number": part_number,
        "invoice_number": invoice_number,
        "customer_po": customer_po
    }


def create_chunks(folder_path: Path) -> List[Dict]:
    chunks = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith((".tif", ".tiff")):
            continue

        try:
            file_path = Path(folder_path) / file_name

            raw_text = run_ocr_tif(file_path)
            clean_chunk = clean_text(raw_text)

            if not clean_chunk:
                print(f"Texte vide ignoré : {file_name}")
                continue

            extracted = extract_metadata(clean_chunk)

            doc_id = file_path.stem
            chunk_id = f"{doc_id}_page_1"

            chunk = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source_file": file_name,
                "page": 1,
                "supplier": "rockwell",
                "document_type": "shipping_invoice",
                "section": "full_page",

                "part_number": extracted["part_number"],
                "invoice_number": extracted["invoice_number"],
                "customer_po": extracted["customer_po"],

                "text": clean_chunk,
                "parent_text": clean_chunk
            }

            chunks.append(chunk)

        except Exception as e:
            print(f"Erreur fichier {file_name}: {e}")

    return chunks


def create_rag_index(
    client: OpenSearch,
    index_name: str = "rockwell_rag",
    embedding_dim: int = 1536,
    recreate: bool = False
):
    if client.indices.exists(index=index_name):
        if recreate:
            client.indices.delete(index=index_name)
            print(f"Index supprimé : {index_name}")
        else:
            print(f"Index existe déjà : {index_name}")
            return

    mapping = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "page": {"type": "integer"},

                "supplier": {"type": "keyword"},
                "document_type": {"type": "keyword"},
                "section": {"type": "keyword"},

                "part_number": {"type": "keyword"},
                "invoice_number": {"type": "keyword"},
                "customer_po": {"type": "keyword"},

                "text": {"type": "text"},
                "parent_text": {"type": "text"},

                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene"
                    }
                }
            }
        }
    }

    client.indices.create(index=index_name, body=mapping)
    print(f"Index créé : {index_name}")


def embed_text(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def ingest_chunks(chunks: List[Dict]):
    for chunk in chunks:
        embedding = embed_text(chunk["text"])
        chunk["embedding"] = embedding

        os_client.index(
            index=INDEX_NAME,
            id=chunk["chunk_id"],
            body=chunk
        )

        print(f"Indexé : {chunk['chunk_id']}")

    os_client.indices.refresh(index=INDEX_NAME)
    print(f"Ingestion terminée : {len(chunks)} chunks")


if __name__ == "__main__":
    create_rag_index(
        client=os_client,
        index_name=INDEX_NAME,
        embedding_dim=1536,
        recreate=True
    )

    chunks = create_chunks(DATA_DIR)
    ingest_chunks(chunks)

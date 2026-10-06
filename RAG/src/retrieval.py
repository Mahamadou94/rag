import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from opensearchpy import OpenSearch

load_dotenv()

INDEX_NAME = "rockwell_rag"
PIPELINE_NAME = "hybrid_pipeline"
EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI(api_key=os.getenv("ORGA_OPENAI_API_KEY"))

os_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False,
    verify_certs=False
)


def embed_text(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def create_hybrid_pipeline(client: OpenSearch) -> None:
    """
    Pipeline OpenSearch native hybrid search :
    BM25 + vector search avec normalisation des scores.
    """

    body = {
        "description": "Hybrid BM25 + vector search pipeline",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {
                        "technique": "min_max"
                    },
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [0.4, 0.6]
                        }
                    }
                }
            }
        ]
    }

    client.transport.perform_request(
        method="PUT",
        url=f"/_search/pipeline/{PIPELINE_NAME}",
        body=body
    )

    print(f"Pipeline créé : {PIPELINE_NAME}")


def native_hybrid_search(
    query: str,
    k: int = 5,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    Recherche hybride native OpenSearch :
    - BM25 sur text
    - kNN sur embedding
    - fusion via search_pipeline
    """

    query_vector = embed_text(query)

    bm25_query = {
        "match": {
            "text": {
                "query": query,
                "boost": 1.0
            }
        }
    }

    vector_query = {
        "knn": {
            "embedding": {
                "vector": query_vector,
                "k": 10
            }
        }
    }

    hybrid_query = {
        "hybrid": {
            "queries": [
                bm25_query,
                vector_query
            ]
        }
    }

    if filters:
        final_query = {
            "bool": {
                "must": [hybrid_query],
                "filter": filters
            }
        }
    else:
        final_query = hybrid_query

    body = {
        "size": k,
        "_source": [
            "chunk_id",
            "doc_id",
            "source_file",
            "page",
            "section",
            "supplier",
            "document_type",
            "part_number",
            "invoice_number",
            "customer_po",
            "text",
            "parent_text"
        ],
        "query": final_query
    }

    response = os_client.search(
        index=INDEX_NAME,
        body=body,
        params={"search_pipeline": PIPELINE_NAME}
    )

    return response["hits"]["hits"]


def bm25_search(query: str, k: int = 5) -> List[Dict]:
    """
    Recherche lexicale simple pour debug.
    """

    body = {
        "size": k,
        "_source": [
            "chunk_id",
            "doc_id",
            "source_file",
            "page",
            "part_number",
            "invoice_number",
            "customer_po",
            "text"
        ],
        "query": {
            "match": {
                "text": query
            }
        }
    }

    response = os_client.search(index=INDEX_NAME, body=body)
    return response["hits"]["hits"]


def vector_search(query: str, k: int = 5) -> List[Dict]:
    """
    Recherche vectorielle simple pour debug.
    """

    query_vector = embed_text(query)

    body = {
        "size": k,
        "_source": [
            "chunk_id",
            "doc_id",
            "source_file",
            "page",
            "part_number",
            "invoice_number",
            "customer_po",
            "text"
        ],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k
                }
            }
        }
    }

    response = os_client.search(index=INDEX_NAME, body=body)
    return response["hits"]["hits"]


def search_by_doc_id(doc_id: str, k: int = 10) -> List[Dict]:
    """
    Voir tous les chunks d'un document.
    """

    body = {
        "size": k,
        "_source": [
            "chunk_id",
            "doc_id",
            "source_file",
            "page",
            "part_number",
            "invoice_number",
            "customer_po",
            "text"
        ],
        "query": {
            "term": {
                "doc_id": doc_id
            }
        }
    }

    response = os_client.search(index=INDEX_NAME, body=body)
    return response["hits"]["hits"]


def print_results(results: List[Dict]) -> None:
    for hit in results:
        src = hit["_source"]

        print("\n==============================")
        print("Score:", hit.get("_score"))
        print("Chunk ID:", src.get("chunk_id"))
        print("Doc ID:", src.get("doc_id"))
        print("Source:", src.get("source_file"))
        print("Page:", src.get("page"))
        print("Part number:", src.get("part_number"))
        print("Invoice:", src.get("invoice_number"))
        print("Customer PO:", src.get("customer_po"))
        print("Text:", src.get("text", "")[:800])


if __name__ == "__main__":
    # À lancer une seule fois si le pipeline n'existe pas encore
    create_hybrid_pipeline(os_client)

    query = "Quelle est la date de la facture pour le part number 822-1821-001 ?"

    results = native_hybrid_search(
        query=query,
        k=5
    )

    print_results(results)

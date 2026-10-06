import os
from dotenv import load_dotenv
from openai import OpenAI

from retrieval import native_hybrid_search

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("ORGA_OPENAI_API_KEY"))


def build_context(results):
    contexts = []

    for i, hit in enumerate(results, start=1):
        src = hit["_source"]

        context = f"""
[Source {i}]
Fichier: {src.get("source_file")}
Doc ID: {src.get("doc_id")}
Page: {src.get("page")}
Part number: {src.get("part_number", "")}
Invoice number: {src.get("invoice_number", "")}
Customer PO: {src.get("customer_po", "")}

Texte OCR:
{src.get("text", "")}
"""
        contexts.append(context)

    return "\n\n".join(contexts)


def answer_question(question: str):
    results = native_hybrid_search(
        query=question,
        k=5
    )

    context = build_context(results)

    prompt = f"""
Tu es un assistant spécialisé dans l'analyse de factures Rockwell Collins.

Tu dois répondre uniquement à partir du contexte OCR fourni.
Si l'information demandée n'est pas présente dans le contexte, réponds :
"Information non trouvée dans les documents récupérés."

Ne devine jamais.
Donne une réponse courte et cite toujours le fichier source et la page.
Exemple : Source : 005=20220817_14335005.tif, page 1.

Question utilisateur :
{question}

Contexte OCR récupéré :
{context}

Réponse :
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    question = "Quelle est le nom du fournisseur et client ainsi que leurs localisation pour le part number 822-1821-001 ?"
    answer = answer_question(question)
    print(answer)

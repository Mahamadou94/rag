# RAG Lab

Ce projet compare deux approches de retrieval pour un pipeline RAG:

- `approaches/qdrant`: approche existante basée sur Qdrant, LangChain et CrewAI.
- `approaches/opensearch`: approche OpenSearch que l'on va implémenter progressivement.

Les dépendances Python communes sont déclarées à la racine dans `requirements.txt` et `pyproject.toml`.

## Environnement

```bash
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

## OpenSearch local

```bash
docker compose -f approaches/opensearch/docker-compose.yml up -d
```

OpenSearch sera disponible sur `http://localhost:9200` et Dashboards sur `http://localhost:5601`.

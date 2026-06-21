# Neo4j Graph Build Extension — Deliverable 2

## Added files

- `build_graph.py`: loads paper metadata from CSV into Neo4j.
- `cypher_queries.cypher`: five example Neo4j queries for evidence and screenshots.

## Graph schema

```text
(:Author)-[:WROTE]->(:Paper)
(:Paper)-[:ABOUT]->(:Topic)
(:Paper)-[:PUBLISHED_IN]->(:Venue)
```

## Required CSV columns

```text
paper id, title, authors, venue, year, topics
```

Example format:

```csv
paper id,title,authors,venue,year,topics
P001,Attention Is All You Need,"Vaswani; Shazeer; Parmar",NeurIPS,2017,"Transformers; NLP"
```

Use semicolons between multiple authors and between multiple topics.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set Neo4j environment variables:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your_password"
export NEO4J_DATABASE="neo4j"
```

For Windows PowerShell:

```powershell
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USERNAME="neo4j"
$env:NEO4J_PASSWORD="your_password"
$env:NEO4J_DATABASE="neo4j"
```

## Run

Validate the CSV without connecting to Neo4j:

```bash
python build_graph.py --csv papers.csv --dry_run
```

Create the graph:

```bash
python build_graph.py --csv papers.csv
```

After graph creation, run the queries in `cypher_queries.cypher` in Neo4j Browser and capture result screenshots for the Deliverable 2 graph evidence section.

// CSAI415 Deliverable 2 — Example Neo4j Cypher Queries
// Graph model:
// (:Author)-[:WROTE]->(:Paper)
// (:Paper)-[:ABOUT]->(:Topic)
// (:Paper)-[:PUBLISHED_IN]->(:Venue)

// Query 1: Find all papers written by a specific author.
// Replace "Vaswani" with an author name that exists in the CSV.
MATCH (a:Author)-[:WROTE]->(p:Paper)
WHERE toLower(a.name) = toLower("Vaswani")
RETURN a.name AS author, p.paper_id AS paper_id, p.title AS paper_title, p.year AS year
ORDER BY p.year DESC;

// Query 2: Find all papers about a specific topic.
// Replace "Transformers" with another stored topic.
MATCH (p:Paper)-[:ABOUT]->(t:Topic)
WHERE toLower(t.name) = toLower("Transformers")
RETURN t.name AS topic, p.paper_id AS paper_id, p.title AS paper_title, p.year AS year
ORDER BY p.year DESC;

// Query 3: Find all papers published in a specific venue.
// Replace "NeurIPS" with another stored venue.
MATCH (p:Paper)-[:PUBLISHED_IN]->(v:Venue)
WHERE toLower(v.name) = toLower("NeurIPS")
RETURN v.name AS venue, p.paper_id AS paper_id, p.title AS paper_title, p.year AS year
ORDER BY p.year DESC;

// Query 4: Find authors who wrote papers about a specific topic.
// Demonstrates a two-hop graph traversal: Author -> Paper -> Topic.
MATCH (a:Author)-[:WROTE]->(p:Paper)-[:ABOUT]->(t:Topic)
WHERE toLower(t.name) = toLower("Transformers")
RETURN DISTINCT a.name AS author, p.title AS paper_title, t.name AS topic
ORDER BY author, paper_title;

// Query 5: Find the ten most common topics in the corpus.
// Counts papers connected to each Topic node.
MATCH (p:Paper)-[:ABOUT]->(t:Topic)
RETURN t.name AS topic, count(p) AS number_of_papers
ORDER BY number_of_papers DESC, topic
LIMIT 10;

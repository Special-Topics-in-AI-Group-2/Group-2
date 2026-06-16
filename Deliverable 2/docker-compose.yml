services:

  # ── MongoDB ───────────────────────────────────────────────────────────────
  # Used by ingest.py (csai415.chunks + csai415.documents)
  # and retriever.py (MONGO_DB defaults to "papers_ai_agent" — override via env)
  mongodb:
    image: mongo:7
    container_name: csai415_mongo
    restart: unless-stopped
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    environment:
      MONGO_INITDB_DATABASE: csai415
    healthcheck:
      test: ["CMD", "mongosh", "--quiet", "--eval", "db.adminCommand('ping').ok"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ── Qdrant ────────────────────────────────────────────────────────────────
  # Used by ingest.py and retriever.py for dense vector search.
  # REST API on 6333, gRPC on 6334.
  qdrant:
    image: qdrant/qdrant:v1.9.2
    container_name: csai415_qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"   # REST / dashboard  →  http://localhost:6333/dashboard
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ── Neo4j ─────────────────────────────────────────────────────────────────
  # Used by build_graph.py.
  # Browser UI on 7474  →  http://localhost:7474
  # Bolt on 7687 (NEO4J_URI=bolt://localhost:7687)
  neo4j:
    image: neo4j:5
    container_name: csai415_neo4j
    restart: unless-stopped
    ports:
      - "7474:7474"   # HTTP browser
      - "7687:7687"   # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    environment:
      NEO4J_AUTH: neo4j/csai415pass        # username: neo4j  password: csai415pass
      NEO4J_PLUGINS: '["apoc"]'            # APOC plugin — useful for graph queries
      NEO4J_dbms_memory_heap_initial__size: 256m
      NEO4J_dbms_memory_heap_max__size:    512m
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "csai415pass", "RETURN 1"]
      interval: 15s
      timeout: 10s
      retries: 10
      start_period: 30s   # Neo4j is slow to boot — give it time

volumes:
  mongo_data:
  qdrant_data:
  neo4j_data:
  neo4j_logs:

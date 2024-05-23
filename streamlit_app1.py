"""
This uses csv instead of dump
"""
import os
import streamlit as st
import time
from langchain.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_groq import ChatGroq
import textwrap

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
# NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


graph = Neo4jGraph(url=NEO4J_URI,
                    username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
WITH row,
    split(row.director, '|') AS directors,
    split(row.actors, '|') AS actors,
    split(row.genres, '|') AS genres
MERGE (m:Movie {id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in directors | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in actors | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in genres | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

# Execute the Cypher query to load data
graph.query(movies_query)
graph.refresh_schema()
success = st.success("Connected to Neo4j")
time.sleep(1)
success.empty()


# Define the Cypher generation template and prompt
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:

# What investment firms are in San Francisco?
MATCH (mgr:Manager)-[:LOCATED_AT]->(mgrAddress:Address)
    WHERE mgrAddress.city = 'San Francisco'
RETURN mgr.managerName

# What investment firms are near Santa Clara?
    MATCH (address:Address)
    WHERE address.city = "Santa Clara"
    MATCH (mgr:Manager)-[:LOCATED_AT]->(managerAddress:Address)
    WHERE point.distance(address.location,
        managerAddress.location) < 10000
    RETURN mgr.managerName, mgr.managerAddress

# What does Palo Alto Networks do?
    CALL db.index.fulltext.queryNodes(
        "fullTextCompanyNames",
        "Palo Alto Networks"
        ) YIELD node, score
    WITH node as com
    MATCH (com)-[:FILED]->(f:Form),
    (f)-[s:SECTION]->(c:Chunk)
    WHERE s.f10kItem = "item1"
RETURN c.text

The question is:
{question}"""


CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE
)

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

cypherChain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)

def prettyCypherChain(question: str) -> str:
    response = cypherChain.run(question)
    return textwrap.fill(response, 60)

st.title("Cypher Query Generator")
question = st.text_input("Enter your question:", "Which director has directed both Florence Pernel and Jerzy Stuhr, but not necessarily in the same movie?")

if st.button("Generate Cypher Query"):
    result = prettyCypherChain(question)
    st.text_area("Generated Cypher Query:", result)

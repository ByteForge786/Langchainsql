from langchain.agents import AgentType, Tool, initialize_agent
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain.schema import BaseLanguageModel
from langchain.chains import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from typing import Any, List, Optional
import json

class CustomLLM(BaseLanguageModel):
    def __call__(self, prompt: str) -> str:
        """Call the custom LLM with get_llm_response"""
        return get_llm_response(prompt)

    def generate(self, prompts: List[str]) -> str:
        """Generate method required by LangChain"""
        return [self(prompt) for prompt in prompts]

    @property
    def _llm_type(self) -> str:
        return "custom"

def create_sql_agent():
    # Initialize our custom LLM
    llm = CustomLLM()
    
    # Create database connection (using SQLite for example)
    db = SQLDatabase.from_uri("sqlite:///your_database.db")
    
    # Initialize the SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create agent with toolkit
    agent = initialize_agent(
        toolkit.get_tools(),
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=ConversationBufferMemory()
    )
    
    return agent

# For schema understanding
def create_schema_agent(schema_path: str):
    # Load schema
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create documents from schema
    docs = []
    for table_name, info in schema['tables'].items():
        content = f"""
        Table: {table_name}
        Description: {info['description']}
        Schema: {info['create_statement']}
        """
        docs.append(Document(page_content=content))
    
    # Create vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Create retrieval tool
    schema_tool = Tool(
        name="schema_lookup",
        func=lambda q: vectorstore.similarity_search(q)[0].page_content,
        description="Search for relevant database schema information"
    )
    
    # Initialize agent with tools
    agent = initialize_agent(
        [schema_tool],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

class SQLQueryAgent:
    def __init__(self, schema_path: str, db_uri: str):
        self.llm = CustomLLM()
        self.db = SQLDatabase.from_uri(db_uri)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Create agents
        self.schema_agent = create_schema_agent(schema_path)
        self.sql_agent = initialize_agent(
            self.toolkit.get_tools(),
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory()
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            # First, understand schema context
            schema_context = self.schema_agent.run(query)
            
            # Then generate and execute SQL
            result = self.sql_agent.run(f"""
            Given this schema context: {schema_context}
            Answer this question: {query}
            """)
            
            return {
                "status": "success",
                "result": result,
                "schema_context": schema_context
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_memory(self):
        self.sql_agent.memory.clear()

# Usage
if __name__ == "__main__":
    agent = SQLQueryAgent(
        schema_path="schema.yaml",
        db_uri="sqlite:///your_database.db"
    )
    
    result = agent.process_query("Show me sales by region")
    print(json.dumps(result, indent=2))

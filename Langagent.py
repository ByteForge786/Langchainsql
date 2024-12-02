///
This implementation:

Uses LangChain's core components:

Tools
Agent
Memory
Prompt Templates
Output Parser


Maintains our custom functionality:

Enhanced schema lookup
SQL generation
Validation
Mock execution
///

  
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import yaml
import json
import logging
import re
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action}\nObservation: {observation}\n"
            
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tools"] = tools_str
        kwargs["thoughts"] = thoughts
        
        return [SystemMessage(content=self.template.format(**kwargs))]

class SchemaLookupTool(Tool):
    def __init__(self, schema_path: str):
        self.name = "schema_lookup"
        self.description = "Find relevant database tables based on the query"
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        with open(schema_path) as f:
            self.schema_config = yaml.safe_load(f)
        self._init_embeddings()
        
    def _init_embeddings(self):
        self.table_embeddings = {}
        for table_name, info in self.schema_config['tables'].items():
            table_text = f"""
            Table: {table_name}
            Description: {info['description']}
            Columns: {self._extract_columns(info['create_statement'])}
            Sample Questions: {' '.join(info.get('sample_questions', []))}
            """
            self.table_embeddings[table_name] = {
                'embedding': self.embed_model.encode(table_text),
                'info': info
            }
    
    def _extract_columns(self, create_statement: str) -> str:
        lines = create_statement.split('\n')
        columns = []
        for line in lines:
            if 'CREATE' not in line and 'PRIMARY KEY' not in line:
                col = line.strip().split(' ')
                if len(col) >= 2:
                    columns.append(f"{col[0]} ({col[1]})")
        return ', '.join(columns)

    def run(self, query: str) -> str:
        logger.info(f"Looking up schema for: {query}")
        try:
            query_embedding = self.embed_model.encode(query)
            relevant_tables = []
            
            for table_name, table_data in self.table_embeddings.items():
                similarity = np.dot(query_embedding, table_data['embedding'])
                if similarity > 0.5:
                    table_info = table_data['info']
                    relevant_tables.append({
                        "name": table_name,
                        "create_statement": table_info['create_statement'],
                        "description": table_info['description']
                    })
            
            result = {
                "relevant_tables": relevant_tables,
                "schema_context": "\n".join(t["create_statement"] for t in relevant_tables)
            }
            return json.dumps(result)
        except Exception as e:
            return str(e)

class SQLGenerationTool(Tool):
    def __init__(self):
        self.name = "sql_generation"
        self.description = "Generate SQL query based on the question and schema"
    
    def run(self, query: str, schema_context: str) -> str:
        prompt = f"""
Generate a SQL query based on:

Question: {query}

Schema:
{schema_context}

Return in this exact format:
{{
    "sql": "the complete SQL query",
    "explanation": "step-by-step explanation of the query logic"
}}
"""
        result = get_llm_response(prompt)
        return json.dumps(result)

class SQLValidationTool(Tool):
    def __init__(self):
        self.name = "sql_validation"
        self.description = "Validate SQL query for safety and correctness"
    
    def run(self, sql: str, schema_context: str) -> str:
        prompt = f"""
Validate this SQL query:
{sql}

Against schema:
{schema_context}

Return in this exact format:
{{
    "is_safe": boolean,
    "issues": ["list of issues found"],
    "feedback": "improvement suggestions if any"
}}
"""
        result = get_llm_response(prompt)
        return json.dumps(result)

class MockDBTool(Tool):
    def __init__(self):
        self.name = "db_execution"
        self.description = "Execute SQL query (mock data for demonstration)"
    
    def run(self, sql: str) -> str:
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'sales': np.random.uniform(1000, 5000, 30),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 30)
        })
        
        result = {
            "data": df.to_dict(),
            "metadata": {
                "row_count": len(df),
                "columns": list(df.columns)
            }
        }
        return json.dumps(result)

class CustomOutputParser:
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        try:
            response = get_llm_response(llm_output)
            
            if response.get("finish", False):
                return AgentFinish(
                    return_values={"output": response.get("output", "Task completed")},
                    log=llm_output,
                )
                
            action = response.get("action", "")
            action_input = response.get("action_input", {})
            
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
            
        except Exception as e:
            logger.error(f"Error parsing output: {str(e)}")
            return AgentFinish(
                return_values={"output": "Error occurred in processing"},
                log=llm_output,
            )

class SQLAgent:
    def __init__(self, schema_path: str):
        # Initialize tools
        self.tools = [
            SchemaLookupTool(schema_path),
            SQLGenerationTool(),
            SQLValidationTool(),
            MockDBTool()
        ]
        
        # Create prompt template
        prompt = CustomPromptTemplate(
            template="""
You are a SQL expert assistant. Given the following context and tools, determine the next action.

Question: {input}
Available Tools: {tools}

Previous Steps:
{thoughts}

Return response in this format:
{{
    "action": "tool_name",
    "action_input": {{
        "parameter": "value"
    }},
    "finish": boolean,
    "output": "final response if finished"
}}
""",
            tools=self.tools,
            input_variables=["input", "intermediate_steps"]
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory()
        
        # Create the agent
        self.agent = LLMSingleActionAgent(
            llm=None,  # We'll use our custom LLM
            prompt=prompt,
            tools=self.tools,
            output_parser=CustomOutputParser()
        )
        
        # Create the executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
    
    def run(self, query: str) -> Dict[str, Any]:
        try:
            result = self.agent_executor.run(query)
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_memory(self):
        self.memory.clear()

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = SQLAgent("schema.yaml")
    
    # Example query
    query = "Show me daily sales trends by region"
    result = agent.run(query)
    
    # Print results
    print(json.dumps(result, indent=2))

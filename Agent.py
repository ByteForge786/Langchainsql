import yaml
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Tool:
    name: str
    description: str
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class SchemaLookupTool(Tool):
    def __init__(self, schema_path: str):
        super().__init__(
            name="schema_lookup",
            description="Find relevant database tables and their relationships based on the query"
        )
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

    def _analyze_query_needs(self, query: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze this query: "{query}"
        
        Determine the data requirements.
        Return in this exact format:
        {{
            "required_data": ["list of data elements needed"],
            "temporal": boolean if time-based analysis needed,
            "aggregation": boolean if aggregation needed,
            "joins_needed": boolean if multiple tables likely needed,
            "explanation": "why these requirements were identified"
        }}
        """
        return get_llm_response(prompt)

    def _get_related_tables(self, table_name: str) -> List[str]:
        related = []
        create_stmt = self.schema_config['tables'][table_name]['create_statement']
        for other_table in self.schema_config['tables']:
            if other_table != table_name:
                if other_table.lower() in create_stmt.lower() or \
                   table_name.lower() in self.schema_config['tables'][other_table]['create_statement'].lower():
                    related.append(other_table)
        return related

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # 1. Analyze query requirements
            query_needs = self._analyze_query_needs(query)
            logger.info(f"Query needs: {query_needs}")
            
            # 2. Get embeddings similarity
            query_embedding = self.embed_model.encode(query)
            relevant_tables = {}
            
            for table_name, table_data in self.table_embeddings.items():
                similarity = np.dot(query_embedding, table_data['embedding'])
                relevant_tables[table_name] = {
                    'similarity': float(similarity),
                    'info': table_data['info']
                }
            
            # 3. Categorize tables
            primary_tables = []
            secondary_tables = []
            
            for table_name, table_info in relevant_tables.items():
                if table_info['similarity'] > 0.6:
                    primary_tables.append(table_name)
                elif table_info['similarity'] > 0.4:
                    secondary_tables.append(table_name)
            
            # 4. Add related tables if joins needed
            if query_needs.get('joins_needed', False):
                for primary_table in primary_tables.copy():
                    related = self._get_related_tables(primary_table)
                    for related_table in related:
                        if related_table not in primary_tables:
                            secondary_tables.append(related_table)

            # 5. Build final table list
            final_tables = []
            seen_tables = set()
            
            for table_name in primary_tables + secondary_tables:
                if table_name not in seen_tables:
                    table_info = self.schema_config['tables'][table_name]
                    final_tables.append({
                        "name": table_name,
                        "create_statement": table_info['create_statement'],
                        "description": table_info['description'],
                        "columns": self._extract_columns(table_info['create_statement']),
                        "relevance": "primary" if table_name in primary_tables else "secondary"
                    })
                    seen_tables.add(table_name)

            logger.info(f"Selected tables: Primary={primary_tables}, Secondary={secondary_tables}")
            
            return {
                "status": "success",
                "relevant_tables": final_tables,
                "schema_context": "\n".join(t["create_statement"] for t in final_tables),
                "query_needs": query_needs,
                "table_relevance": {
                    "primary": primary_tables,
                    "secondary": secondary_tables
                }
            }
            
        except Exception as e:
            logger.error(f"Schema lookup error: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e)
            }

class SQLGenerationTool(Tool):
    def __init__(self):
        super().__init__(
            name="sql_generation",
            description="Generate SQL query based on the question and schema"
        )
    
    def execute(self, question: str, schema_context: str, query_needs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
Generate a SQL query based on:

Question: {question}

Schema:
{schema_context}

Query Requirements:
{json.dumps(query_needs, indent=2)}

Return in this exact format:
{{
    "status": "success",
    "sql": "the complete SQL query",
    "explanation": "step-by-step explanation of the query logic",
    "tables_used": ["list of tables used in query"],
    "columns_used": ["list of columns used in query"]
}}
"""
        return get_llm_response(prompt)

class SQLValidationTool(Tool):
    def __init__(self):
        super().__init__(
            name="sql_validation",
            description="Validate SQL query for safety and correctness"
        )
    
    def execute(self, sql: str, schema_context: str, tables_used: List[str]) -> Dict[str, Any]:
        prompt = f"""
Validate this SQL query:
{sql}

Against schema:
{schema_context}

Tables being used:
{json.dumps(tables_used, indent=2)}

Check for:
1. SQL injection risks
2. Proper column references
3. Join conditions
4. Where clause correctness
5. Group by completeness
6. Overall syntax

Return in this exact format:
{{
    "status": "success" or "error",
    "is_safe": boolean,
    "issues": ["list of specific issues found"],
    "feedback": "detailed suggestions for improvement",
    "join_analysis": "analysis of join conditions",
    "where_analysis": "analysis of where conditions"
}}
"""
        return get_llm_response(prompt)

class MockDBTool(Tool):
    def __init__(self):
        super().__init__(
            name="db_execution",
            description="Execute SQL query (mock data for demonstration)"
        )
    
    def execute(self, sql: str, tables_used: List[str]) -> Dict[str, Any]:
        try:
            # Generate mock data based on tables being used
            df = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=30),
                'sales': np.random.uniform(1000, 5000, 30),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 30),
                'product': np.random.choice(['A', 'B', 'C'], 30),
                'customer': np.random.choice(['X', 'Y', 'Z'], 30)
            })
            
            return {
                "status": "success",
                "data": df,
                "metadata": {
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "tables_used": tables_used
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

class SQLReActAgent:
    def __init__(self, schema_path: str):
        self.tools = {
            "schema_lookup": SchemaLookupTool(schema_path),
            "sql_generation": SQLGenerationTool(),
            "sql_validation": SQLValidationTool(),
            "db_execution": MockDBTool()
        }
        self.memory = []
    
    def get_next_action(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine next action using ReAct prompt"""
        tools_desc = "\n".join(f"- {name}: {tool.description}" 
                              for name, tool in self.tools.items())
        
        prompt = f"""
You are a SQL agent. Determine the next action based on the current state.

User Question: {query}

Available Tools:
{tools_desc}

Current Context:
{json.dumps(context, indent=2)}

Previous Actions:
{json.dumps(self.memory, indent=2)}

Think carefully about what's needed next.
Return in this exact format:
{{
    "thought": "your step-by-step reasoning about what to do next",
    "action": "tool_name to use",
    "action_input": {{
        "required parameters for the tool"
    }},
    "should_continue": boolean
}}
"""
        return get_llm_response(prompt)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main ReAct loop for processing query"""
        context = {"query": query}
        executed_actions = []
        
        try:
            while True:
                # Get next action
                action_plan = self.get_next_action(query, context)
                logger.info(f"Next action: {action_plan}")
                
                if not action_plan.get("should_continue", False):
                    break
                
                tool_name = action_plan.get("action")
                if tool_name not in self.tools:
                    logger.error(f"Unknown tool: {tool_name}")
                    continue
                
                # Execute tool
                tool = self.tools[tool_name]
                try:
                    result = tool.execute(**action_plan.get("action_input", {}))
                    
                    # Record action
                    executed_action = {
                        "tool": tool_name,
                        "thought": action_plan.get("thought"),
                        "status": result.get("status", "error"),
                        "result": result
                    }
                    executed_actions.append(executed_action)
                    self.memory.append(executed_action)
                    
                    # Update context
                    if result["status"] == "success":
                        context.update(result)
                    else:
                        logger.error(f"Tool execution failed: {result}")
                        break
                        
                except Exception as e:
                    logger.error(f"Error executing {tool_name}: {traceback.format_exc()}")
                    break
            
            return {
                "type": "sql_analysis",
                "context": context,
                "actions": executed_actions,
                "status": "success" if executed_actions else "error"
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {traceback.format_exc()}")
            return {
                "type": "error",
                "message": str(e),
                "status": "error"
            }

    def clear_memory(self):
        """Clear agent's memory"""
        self.memory = []

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = SQLReActAgent("schema.yaml")
    
    # Process a query
    query = "Show me daily sales trends by region and product category"
    result = agent.process_query(query)
    
    # Print results
    if result["status"] == "success":
        print("Relevant Tables:", result["context"].get("table_relevance", {}))
        print("Generated SQL:", result["context"].get("sql", ""))
        print("Execution Results:", result["context"].get("metadata", {}))
    else:
        print("Error:", result.get("message", "Unknown error"))

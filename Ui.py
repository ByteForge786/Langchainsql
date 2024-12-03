import streamlit as st
import json
from typing import Dict, Any
import pandas as pd
import yaml
from sql_agent import SQLReActAgent  # Assuming your agent code is in sql_agent.py

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="SQL Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stAlert {
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .sql-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 4px;
        font-family: monospace;
    }
    .thought-box {
        background-color: #e8f0fe;
        padding: 10px;
        border-radius: 4px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

def display_sql_analysis(result: Dict[str, Any]):
    """Display SQL analysis results in a structured format"""
    if result.get("status") == "success":
        for action in result.get("actions", []):
            with st.expander(f"🔍 {action['tool'].replace('_', ' ').title()}", expanded=True):
                # Display thought process
                if action.get("thought"):
                    st.markdown("**Thought Process:**")
                    st.markdown(f"""<div class="thought-box">{action['thought']}</div>""", 
                              unsafe_allow_html=True)

                # Display tool-specific results
                if action["tool"] == "schema_lookup":
                    if action["result"].get("relevant_tables"):
                        st.markdown("**Relevant Tables:**")
                        for table in action["result"]["relevant_tables"]:
                            st.markdown(f"""
                            - **{table['name']}** ({table['relevance']})
                            - Description: {table['description']}
                            - Columns: {table['columns']}
                            """)

                elif action["tool"] == "sql_generation":
                    if action["result"].get("sql"):
                        st.markdown("**Generated SQL:**")
                        st.code(action["result"]["sql"], language="sql")
                        st.markdown("**Explanation:**")
                        st.write(action["result"].get("explanation", ""))

                elif action["tool"] == "sql_validation":
                    if action["result"].get("is_safe") is not None:
                        st.markdown("**Validation Results:**")
                        status_color = "green" if action["result"]["is_safe"] else "red"
                        st.markdown(f"""
                        - Status: <span style='color: {status_color}'>
                            {'✅ Safe' if action["result"]["is_safe"] else '❌ Issues Found'}
                        </span>
                        """, unsafe_allow_html=True)
                        if action["result"].get("issues"):
                            st.markdown("**Issues Found:**")
                            for issue in action["result"]["issues"]:
                                st.markdown(f"- {issue}")
                        if action["result"].get("feedback"):
                            st.markdown("**Suggestions:**")
                            st.write(action["result"]["feedback"])

                elif action["tool"] == "db_execution":
                    if action["result"].get("data") is not None:
                        st.markdown("**Query Results:**")
                        df = pd.DataFrame(action["result"]["data"])
                        st.dataframe(df, use_container_width=True)
                        st.markdown(f"""
                        **Metadata:**
                        - Rows: {action["result"]["metadata"]["row_count"]}
                        - Columns: {', '.join(action["result"]["metadata"]["columns"])}
                        """)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = SQLReActAgent("schema.yaml")

def main():
    st.title("🤖 SQL Chat Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if message.get("result"):
                    display_sql_analysis(message["result"])
                else:
                    st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your query..."):
                result = st.session_state.agent.process_query(prompt)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "result": result
                })
                
                # Display analysis
                display_sql_analysis(result)
    
    # Add a clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.agent.clear_memory()
        st.rerun()

    # Display schema info in sidebar
    with st.sidebar:
        st.markdown("### Schema Information")
        try:
            with open("schema.yaml", "r") as f:
                schema = yaml.safe_load(f)
                for table_name, info in schema["tables"].items():
                    with st.expander(f"📋 {table_name}"):
                        st.markdown(f"**Description:** {info['description']}")
                        st.markdown("**Sample Questions:**")
                        for q in info.get("sample_questions", []):
                            st.markdown(f"- {q}")
        except Exception as e:
            st.error(f"Error loading schema: {str(e)}")

if __name__ == "__main__":
    main()

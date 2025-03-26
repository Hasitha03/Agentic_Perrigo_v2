"""
ui2.py

Streamlit-based UI for the multi-agent generative AI system.
"""
import os
import time
import uuid
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages

from src.orchestrater.MultiAgentGraph import multi_agent_graph

# ---------------------- Helper Functions ----------------------

def display_saved_plot(plot_path: str):
    """
    Loads and displays a saved plot from the given path in a Streamlit app.

    Args:
        plot_path (str): Path to the saved plot image.
    """
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Generated Plot", use_column_width=True)
    else:
        st.error(f"Plot not found at {plot_path}")

def reset_app_state():
    """Reset the app state when the data source changes."""
    st.session_state.initialized = False
    st.session_state.pop('df', None)

def load_data_file(filename):
    """Load a CSV file with automatic date parsing."""
    try:
        date_columns = [col for col in pd.read_csv(filename, nrows=1).columns if 'date' in col.lower()]
        return pd.read_csv(filename, parse_dates=date_columns, dayfirst=True)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

# ---------------------- Sidebar Setup ----------------------

def setup_sidebar():
    """Set up sidebar with API key input and data source selection."""

    api_key = st.session_state["OPENAI_API_KEY"]
    Outbound_data = os.path.join("src", "data", "Outbound_Data.csv")
    Inventory_Data = os.path.join("src", "data", "Inventory_Batch.csv")
    Inbound_Data = os.path.join("src", "data", "Outbound_Data.csv")
    data_files = {
        'Outbound_Data.csv': Outbound_data,
        'Inventory_Batch.csv': Inventory_Data,
        'Inbound_Data.csv': Inbound_Data
    }

    st.sidebar.subheader("📂 Select Data Source")
    data_source = st.sidebar.radio("Choose Data Source:", list(data_files.keys()), index=0)

    if st.session_state.get('current_data_source') != data_source:
        st.session_state.current_data_source = data_source
        reset_app_state()

    return api_key, data_files[data_source]

# ---------------------- UI Components ----------------------

def display_sample_data():
    """Display sample data in an expander."""
    with st.expander("📊 View Sample Data"):
        df = st.session_state.df.copy()
        for col in df.select_dtypes(include=['datetime64']):
            df[col] = df[col].dt.strftime('%d-%m-%Y')
        st.dataframe(df.head(), use_container_width=True)

# ---------------------- Main UI Function ----------------------

def main():
    """Main UI function to handle user interactions and execute the multi-agent graph."""

    api_key, data_file = setup_sidebar()
    os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        st.info("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    if 'df' not in st.session_state:
        st.session_state.df = load_data_file(data_file)
        if st.session_state.df is None:
            st.stop()

    display_sample_data()


    if "bi_agent_responses" not in st.session_state:
        st.session_state.bi_agent_responses = []
    if 'cost_optimization_response' not in st.session_state:
        st.session_state.cost_optimization_response = []
    if 'static_optimization_response' not in st.session_state:
        st.session_state.static_optimization_response = []

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("GenAI Answer Bot")
        question = st.text_input("Please ask your question:", key="beginning_query")
        if st.button("Submit", key="langgraph_chat_submit"):
            if not question:
                st.error("Please enter a question before submitting.")
                return

            # Start multi-agent graph processing
            graph_start_time = time.time()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            state = {"messages": [HumanMessage(content=question)], "next": "supervisor"}

            counter = 0
            while state['next'] != 'FINISH' and counter < 10:
                node_time = time.time()
                current_state = multi_agent_graph.nodes[state['next']].invoke(state, config)

                # Display results in a styled container
                st.markdown(f"""
                        <div style="background-color: #eaecee; padding: 10px; border-radius: 10px; margin: 10px 0;">
                            <strong style="color: #2a52be;">{state['next'].upper()}:</strong>
                            <p style="color: #333;">{current_state['messages'][0].content}</p>
                        </div>
                    """, unsafe_allow_html=True)

                state['messages'] = add_messages(state['messages'], current_state['messages'])
                state['next'] = current_state['next']
                counter += 1
                # st.info(f"Node analysis completed in {time.time() - node_time:.1f} seconds")

            # Display completion message
            st.markdown(
                f"""
                    <div style="
                        background-color:#262730;
                        color:#00ffcc;
                        padding:10px;
                        border-radius:8px;
                        font-size:16px;
                        text-align:center;
                    ">
                        ⏱️ <b>Analysis completed in {time.time() - graph_start_time:.1f} seconds</b>
                    </div>
                    """,
                unsafe_allow_html=True
            )
            st.success("Processing complete.")

    with col2:
        st.subheader("🔍 History")
        st.write(" ")
        if st.session_state.bi_agent_responses:
            for i, response in enumerate(st.session_state.bi_agent_responses):
                with st.expander(f"Query: {response['question'][:30]}...", expanded=False):
                    st.markdown(f"**Question:** {response['question']}")
                    st.markdown(f"**Time:** {response['timestamp']}")
                    st.markdown("**Answer:**")
                    st.markdown(response['answer'])

                    if response['figure']:
                        st.image(response['figure'])

        if st.session_state.cost_optimization_response:
            for i, response in enumerate(st.session_state.cost_optimization_response):
                with st.expander(f"Query: {response['query'][:30]}...", expanded=False):
                    st.markdown(f"**Question:** {response['query']}")
                    st.markdown(f"**Time:** {response['timestamp']}")
                    st.markdown("**Answer:**")
                    st.markdown(response['answer'])

        if st.session_state.static_optimization_response:
            for i, response in enumerate(st.session_state.static_optimization_response):
                with st.expander(f"Query: {response['query'][:30]}...", expanded=False):
                    st.markdown(f"**Question:** {response['query']}")
                    st.markdown(f"**Time:** {response['timestamp']}")
                    st.markdown("**Answer:**")
                    st.markdown(response['answer'])

        if not st.session_state.bi_agent_responses and not st.session_state.cost_optimization_response and not st.session_state.static_optimization_response:
            st.info("No responses yet.")

if __name__ == '__main__':
    main()



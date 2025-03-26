"""
MultiAgentGraph.py

This module defines the multi-agent graph for the generative AI project.
It coordinates different agent nodes (BI Agent, Cost Saving Agent, etc.)
using a supervisor to route the conversation flow.
Prompt templates are loaded from the prompt_templates folder.
"""

import os
import pandas as pd
import streamlit as st
import functools
import warnings
from dotenv import load_dotenv, find_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, AIMessage
from langchain.schema import HumanMessage

from config import display_saved_plot
from src.agents.BIAgent_Node import BIAgent_Class, execute_analysis
from src.orchestrater.supervisor import supervisor_chain, members
from src.agents.CostOptimization_Node import AgenticCostOptimizer
from src.agents.Static_CostOptimization_Node import Static_CostOptimization_Class
from src.utils.openai_api import get_supervisor_llm
from src.utils.load_templates import load_template
warnings.filterwarnings("ignore")

# Load environment variables
_ = load_dotenv(find_dotenv())

llm = get_supervisor_llm()


class AgentState(dict):
    """Defines the structure for passing messages and state transitions."""
    messages: list[AnyMessage]
    next: str


def supervisor_node(state: AgentState):
    """
    Supervisor Node: Uses the supervisor chain to determine the next agent.
    """
    result = supervisor_chain.invoke(state['messages'])
    return {"messages": [AIMessage(content=f"Calling {result['next']}...")], "next": result['next']}

# ---------------------- Generic Agent Node ----------------------

def agent_node(state, agent, name: str):
    """
    Generic agent node that calls the provided agent function with the state.
    """
    result = agent(state)
    return {"messages": result, "next": "supervisor"}

# ---------------------- BI Agent ----------------------

def bi_agent(state: AgentState):
    """
    BI Agent: Responsible for analyzing shipment data and generating insights.
    Uses a dynamically loaded prompt from the prompt_templates folder.
    """
    # Load BI Agent prompt
    bi_prompt = load_template("bi_agent_prompt.txt")

    # Load dataset
    data_path = os.path.join("src", "data", "Outbound_Data.csv")
    df = pd.read_csv(data_path)

    data_description = load_template("Outbound_data.txt")

    # Initialize BI Agent
    agent_instance = BIAgent_Class(
        llm=llm, 
        prompt=bi_prompt, 
        tools=[], 
        data_description=data_description, 
        dataset=df, 
        helper_functions={"execute_analysis": execute_analysis}
    )

    # Extract user question
    question = state['messages'][-2].content if len(state['messages']) >= 2 else state['messages'][-1].content

    response = agent_instance.generate_response(question)

    # cached_response = cache.retrieve_from_cache(question)
    # if cached_response:
    #     st.info("Cache hit!")
    #     response = cached_response['metadatas'][0]
    # else:
    #     # If not in cache, process the question using the agent
    #     response = agent_instance.generate_response(question)
    #
    #     # Update cache with the new question and response
    #     cache.update_cache(question, response)

    if 'bi_agent_responses' not in st.session_state:
        st.session_state.bi_agent_responses = []

        # Create a response object with the question and answer
    bi_response = {
        'question': question,
        'answer': response['answer'],
        'figure': response['figure'],
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    st.session_state.bi_agent_responses.append(bi_response)

    if response['figure']:
        display_saved_plot(response['figure'])
    message = response['answer']

    return [HumanMessage(content=message)]

# ---------------------- Cost Optimization Agent ----------------------

def Dynamic_CostOptimization_Agent(state: AgentState):
    """
    Dynamic Cost Optimization Agent: Analyzes shipment cost-related data and recommends
    strategies to reduce costs.
    """

    file_path = os.path.join("src", "data", "Complete Input.xlsx")
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    query = state['messages'][-2].content if len(state['messages']) >= 2 else state['messages'][-1].content

    parameters = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "query": query,
        "file_name": file_path,
        "df": df
    }

    # Initialize Cost Optimization Agent
    agent_instance = AgenticCostOptimizer(llm, parameters)
    conv = agent_instance.handle_query(query)

    chat=[]
    for msg in conv:
        key, value = list(msg.items())[0]
        if "Agent" in key:
            if type(value) is not str:
                value = str(value)
            chat.append(AIMessage(content=value))
        else:
            chat.append(HumanMessage(content=value))

    result = llm.invoke(f"This is the response provided by the Dynamic Cost Optimization Agent: {chat}. Generate a final response to be shown to the client.")
    message = result.content

    if 'cost_optimization_response' not in st.session_state:
        st.session_state.cost_optimization_response = []

        # Create a response object with the question and answer
    consolidation_response = {
        'query': query,
        'answer':message,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Add to the list of stored responses
    st.session_state.cost_optimization_response.append(consolidation_response)

    return [HumanMessage(content=message)]


# -------------------Static cost optimization Agent --------------------

def Static_CostOptimization_agent(state: AgentState):
    """The Static Cost Optimization Agent is designed to analyze and optimize shipment costs by
    evaluating scenarios before and after consolidation. Using a Rate Card (which includes product type, short postcode, and cost per pallet),
    the agent calculates the base shipment costs. To maximize cost savings, the agent evaluates multiple delivery
    day scenarios (e.g., 5-day, 4-day, or 3-day delivery options).By applying consolidation day mappings, the agent
    aggregates shipments into fewer deliveries, reducing overall costs. The results include: Total shipment costs before and after consolidation ,
    Percentage savings achieved ,Key metrics such as the number of shipments and average pallets per shipment.
    This tool empowers users to identify the most cost-effective delivery strategies while maintaining operational efficiency."""

    # cost_saving_input_df = pd.read_excel('Complete Input.xlsx', sheet_name='Sheet1')
    file_path = os.path.join("src", "data", "Complete Input.xlsx")
    cost_saving_input_df = pd.read_excel(file_path, sheet_name="Sheet1")
    file = os.path.join("src", "data", "Cost per pallet.xlsx")
    rate_card = pd.read_excel(file)

    query = state['messages'][len(state['messages']) - 2].content

    parameters = {"api_key": os.getenv("OPENAI_API_KEY"),
                  "query": query,
                  "complete_input": cost_saving_input_df,
                  "rate_card" : rate_card
                  }
    Static_agent = Static_CostOptimization_Class(llm, parameters)
    conv = Static_agent.handle_question(query)

    chat = []
    for msg in conv:
        key, value = list(msg.items())[0]
        if "Agent" in key:
            if type(value) is not str:
                value = str(value)
            chat.append(AIMessage(content=value))
        else:
            chat.append(HumanMessage(content=value))

    result = llm.invoke(
        f"This is the response provided by the Static Cost Optimization Agent: {chat}. Generate a final response to be shown to the client.")

    message = result.content

    if 'cost_optimization_response' not in st.session_state:
        st.session_state.cost_optimization_response = []

        # Create a response object with the question and answer
    consolidation_response = {
        'query': query,
        'answer':message,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Add to the list of stored responses
    st.session_state.cost_optimization_response.append(consolidation_response)

    return [HumanMessage(content=message)]

# ---------------------- Scenario Agent ----------------------

def generate_scenario_agent(state: AgentState):
    """
    Scenario Agent: Analyzes and compares different operational scenarios.
    """
    message = "Generate Scenario Agent Called..! Choose next as 'FINISH'."
    return [HumanMessage(content=message)]

# ---------------------- Driver Identification Agent ----------------------

def driver_identification_agent(state: AgentState):
    """
    Driver Identification Agent: Identifies the cost drivers for shipments.
    """
    message = "Driver Identification Agent Called..! Choose next as 'FINISH'."
    return [HumanMessage(content=message)]

# ---------------------- Conversation Agent ----------------------

def conversation_agent(state: AgentState):
    """
    Conversation Agent: Acts as a fallback for handling user queries 
    that do not fit into predefined agent categories.
    """
    query = state['messages'][-2].content if len(state['messages']) >= 2 else state['messages'][-1].content
    response = llm.invoke(f"You are a helpful assistant. Answer this: {query}")
    return [HumanMessage(content=response.content)]

# ---------------------- Workflow Setup ----------------------

# Define agent nodes
bi_agent_node = functools.partial(agent_node, agent=bi_agent, name="BI Agent")
driver_identification_agent_node = functools.partial(agent_node, agent=driver_identification_agent, name="Driver Identification Agent")
dynamic_cost_optimization_node = functools.partial(agent_node, agent=Dynamic_CostOptimization_Agent, name="Dynamic Cost Optimization Agent")
static_cost_optimization_node = functools.partial(agent_node , agent= Static_CostOptimization_agent , name ="Static Cost Optimization Agent")
generate_scenario_agent_node = functools.partial(agent_node, agent=generate_scenario_agent, name="Generate Scenario Agent")
conversation_agent_node = functools.partial(agent_node, agent=conversation_agent, name="Conversation Agent")

# Define the multi-agent workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("BI Agent", bi_agent_node)
workflow.add_node("Driver Identification Agent", driver_identification_agent_node)
workflow.add_node("Dynamic Cost Optimization Agent", dynamic_cost_optimization_node)
workflow.add_node("Static Cost Optimization Agent" , static_cost_optimization_node)
workflow.add_node("Generate Scenario Agent", generate_scenario_agent_node)
workflow.add_node("Conversation Agent", conversation_agent_node)
workflow.add_node("supervisor", supervisor_node)

# Supervisor handles routing
for member in members:
    workflow.add_edge(member['agent_name'], "supervisor")

# Define conditional routing
conditional_map = {k['agent_name']: k['agent_name'] for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Set entry point
workflow.set_entry_point("supervisor")

# Compile the workflow
multi_agent_graph = workflow.compile()

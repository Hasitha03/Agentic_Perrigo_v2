## Setting up API key and environment related imports

from dotenv import load_dotenv,find_dotenv
from datetime import datetime, timedelta
import streamlit as st
_ = load_dotenv(find_dotenv())

import pandas as pd


import warnings
warnings.filterwarnings("ignore")

from src.core.order_consolidation.static_consolidation import find_cost_savings
from src.core.order_consolidation.dynamic_consolidation import get_parameters_values


class Static_CostOptimization_Class():
    def __init__(self, llm, parameters):
        """
        Initialize the Agentic Cost Optimizer.

        :param llm: The LLM model to use for queries.
        :param parameters: Dictionary containing required parameters.
        """
        self.llm = llm
        self.parameters = parameters
        self.complete_input = parameters.get("complete_input", pd.DataFrame())
        self.rate_card = parameters.get("rate_card" ,pd.DataFrame())


    def find_cost_savings(self):
        all_results, best_scenario = find_cost_savings(
            self.parameters,
            self.complete_input,
            self.rate_card
        )

        # Update the parameters dictionary with the results
        self.parameters['all_results'] = all_results
        self.parameters['best_scenario'] = best_scenario

    def handle_question(self,question):
        chat_history = [{"Human": question}]

        extracted_params = get_parameters_values(self.parameters["api_key"], question)
        self.parameters.update(extracted_params)
        chat_history.append({"Agent": f"Parameters extracted: {extracted_params}"})

        self.find_cost_savings()

        chat_history.append({"Agent": f"Scenarios of all possible days: {self.parameters['all_results']}"})
        chat_history.append({"Agent": f"Best scenarios for cost savings: {self.parameters['best_scenario']}"})

        return chat_history










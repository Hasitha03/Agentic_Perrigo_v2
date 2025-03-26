"""
CostOptimization_Node.py

This module implements the Cost Optimization Agent. It processes shipment data,
extracts cost optimization parameters, consolidates shipments, computes cost savings,
and generates a summary response.
"""

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import pandas as pd
import streamlit as st

from langchain_experimental.agents import create_pandas_dataframe_agent

from src.core.order_consolidation.dynamic_consolidation import (
    load_data,
    get_parameters_values,
    consolidate_shipments,
    calculate_metrics,
    analyze_consolidation_distribution,
    get_filtered_data,
    agent_wrapper,
    create_shipment_window_vs_saving_plot,
    create_calendar_heatmap_before_vs_after
)

from config.config import  display_saved_plot
from src.utils.openai_api import get_supervisor_llm

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

llm = get_supervisor_llm()


class AgenticCostOptimizer:
    def __init__(self, llm, parameters):
        """
        Initialize the Agentic Cost Optimizer.

        :param llm: The LLM model to use for queries.
        :param parameters: Dictionary containing required parameters.
        """
        self.llm = llm
        self.parameters = parameters
        self.df = parameters.get("df", pd.DataFrame())
        self.shipment_window_range = (1, 10)
        self.total_shipment_capacity = 36
        self.utilization_threshold = 95

    def load_data(self):
        complete_input = os.path.join(os.getcwd() , "src/data/Complete Input.xlsx")
        rate_card_ambient = pd.read_excel(complete_input, sheet_name='AMBIENT')
        rate_card_ambcontrol = pd.read_excel(complete_input, sheet_name='AMBCONTROL')
        return {"rate_card_ambient": rate_card_ambient, "rate_card_ambcontrol": rate_card_ambcontrol}

    def get_filtered_df_from_question(self):
        """Extracts filtered data based on user query parameters."""
        group_field = 'SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME'
        df = self.parameters['df']
        df['SHIPPED_DATE'] = pd.to_datetime(df['SHIPPED_DATE'], dayfirst=True)

        df = get_filtered_data(self.parameters, df)
        if df.empty:
            raise ValueError("No data available for selected parameters. Try again!")
        return df

    def get_cost_saving_data(self):
        """Runs cost-saving algorithm and returns result DataFrame."""

        df = self.get_filtered_df_from_question()
        group_field = 'SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME'

        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])

        best_metrics = None
        best_consolidated_shipments = None
        best_params = None

        all_results = []
        rate_card = load_data()
        for shipment_window in range(self.shipment_window_range[0], self.shipment_window_range[1] + 1):
            st.toast(f"Consolidating orders for shipment window: {shipment_window}")
            high_priority_limit = 0
            all_consolidated_shipments = []
            for _, group_df in grouped:
                consolidated_shipments, _ = consolidate_shipments(
                    group_df, high_priority_limit, self.utilization_threshold, shipment_window, date_range,
                    lambda: None, self.total_shipment_capacity, rate_card
                )
                all_consolidated_shipments.extend(consolidated_shipments)

            metrics = calculate_metrics(all_consolidated_shipments, df)
            distribution, distribution_percentage = analyze_consolidation_distribution(all_consolidated_shipments, df)

            result = {
                'Shipment Window': shipment_window,
                'Total Orders': metrics['Total Orders'],
                'Total Shipments': metrics['Total Shipments'],
                'Total Shipment Cost': round(metrics['Total Shipment Cost'], 1),
                'Total Baseline Cost': round(metrics['Total Baseline Cost'], 1),
                'Cost Savings': metrics['Cost Savings'],
                'Percent Savings': round(metrics['Percent Savings'], 1),
                'Average Utilization': round(metrics['Average Utilization'], 1),
                'CO2 Emission': round(metrics['CO2 Emission'], 1)
            }
            all_results.append(result)

        # Update best results if current combination is better
        if best_metrics is None or metrics['Cost Savings'] > best_metrics['Cost Savings']:
            best_metrics = metrics
            best_consolidated_shipments = all_consolidated_shipments
            best_params = (shipment_window, high_priority_limit, self.utilization_threshold)

        summary_text = (
            f"Optimizing outbound deliveries and identifying cost-saving opportunities involve analyzing various factors "
            f"such as order patterns, delivery routes, shipping costs, and consolidation opportunities.\n\n"

            f"On analyzing the data, I can provide some estimates of cost savings on the historical data if we were to "
            f"group orders to consolidate deliveries.\n\n"

            "**APPROACH TAKEN**\n\n"  # Ensure it's already in uppercase and same.
            f"To consolidate the deliveries, A heuristic approach was used, and the methodology is as follows:\n\n"

            f"**Group Shipments**: Orders are consolidated within a shipment window to reduce transportation costs while "
            f"maintaining timely deliveries. A shipment window represents the number of days prior to the current delivery "
            f"that the order could be potentially shipped, thus representing an opportunity to group it with earlier deliveries.\n\n"

            f"**Iterate Over Shipment Windows**: The model systematically evaluates all possible shipment windows, testing "
            f"different configurations to identify the most effective scheduling approach.\n\n"

            f"**Performance Metric Calculation**: Key performance metrics are assessed for each shipment window, including:\n"
            f"- **Cost savings**\n"
            f"- **Utilization rate**\n"
            f"- **CO2 emission reduction**\n\n"

            f"**Comparison and Selection**: After evaluating all configurations, the shipment window that maximizes cost savings "
            f"while maintaining operational efficiency is identified, and results are displayed as per the best parameter.\n\n"

            f"This method allows us to optimize logistics operations dynamically, ensuring that both financial and environmental "
            f"factors are balanced effectively."
        )
        with st.expander("**VIEW APPROACH OF COST CONSOLIDATION**", expanded=False):
            st.write(summary_text)


        # Updating the parameters with adding shipment window vs cost saving table..
        self.parameters['all_results'] = pd.DataFrame(all_results)
        self.parameters['best_params'] = best_params

    def consolidate_for_shipment_window(self):
        """Runs consolidation algorithm based on the selected shipment window."""
        df = self.get_filtered_df_from_question()
        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])

        rate_card = self.load_data()
        all_consolidated_shipments = []
        for _, group_df in grouped:
            consolidated_shipments, _ = consolidate_shipments(
                group_df, 0, 95, self.parameters['window'], date_range, lambda: None, self.total_shipment_capacity,
                rate_card
            )
            all_consolidated_shipments.extend(consolidated_shipments)

        selected_postcodes = ", ".join(self.parameters["selected_postcodes"]) if self.parameters[
            "selected_postcodes"] else "All Postcodes"
        selected_customers = ", ".join(self.parameters["selected_customers"]) if self.parameters[
            "selected_customers"] else "All Customers"

        metrics = calculate_metrics(all_consolidated_shipments, df)
        main_text = (
            f"Through extensive analysis, the OPTIMAL SHIPMENT WINDOW was determined to be **{self.parameters['best_params'][0]}**, "
            f"with a PALLET SIZE of **46** for **postcodes**: {selected_postcodes} and **customers**: {selected_customers}."
            f"These optimizations resulted in SIGNIFICANT EFFICIENCY IMPROVEMENTS:\n\n"

            f"**SHIPMENT WINDOW**: The most effective shipment window was identified as **{self.parameters['best_params'][0]} DAYS**.\n\n"

            f"**COST SAVINGS**: A reduction of **£{metrics['Cost Savings']:,.1f}**, equating to an **£{metrics['Percent Savings']:.1f}%** decrease in overall transportation costs.\n\n"

            f"**ORDER & SHIPMENT SUMMARY**:\n"
            f"- TOTAL ORDERS PROCESSED: **{metrics['Total Orders']:,}** \n"
            f"- TOTAL SHIPMENTS MADE: **{metrics['Total Shipments']:,}**\n\n"

            f"**UTILIZATION EFFICIENCY**:\n"
            f"- AVERAGE TRUCK UTILIZATION increased to **{metrics['Average Utilization']:.1f}%**, ensuring fewer trucks operate at low capacity.\n\n"

            f"**ENVIRONMENTAL IMPACT**:\n"
            f"- CO2 EMISSIONS REDUCTION: A decrease of **{metrics['CO2 Emission']:,.1f} Kg**, supporting sustainability efforts and reducing the carbon footprint.\n\n"

            f"These optimizations not only lead to substantial COST REDUCTIONS but also enhance OPERATIONAL SUSTAINABILITY, "
            f"allowing logistics operations to function more efficiently while MINIMIZING ENVIRONMENTAL IMPACT."
        )

        with st.expander("**IDENTIFIED COST SAVINGS AND KEY PERFORMANCE INDICATORS(KPIs)**", expanded=False):
            st.write(main_text)



        self.parameters['all_consolidated_shipments'] = pd.DataFrame(all_consolidated_shipments)
        self.parameters['filtered_df'] = df

    def compare_before_and_after_consolidation(self):
        """Compares shipments before and after consolidation."""
        consolidated_df = self.parameters['all_consolidated_shipments']
        df = self.get_filtered_df_from_question()

        before = {
            "Days": df['SHIPPED_DATE'].nunique(),
            "Pallets Per Day": df['Total Pallets'].sum() / df['SHIPPED_DATE'].nunique(),
            "Pallets per Shipment": df['Total Pallets'].sum() / len(df)
        }
        after = {
            "Days": consolidated_df['Date'].nunique(),
            "Pallets Per Day": consolidated_df['Total Pallets'].sum() / consolidated_df['Date'].nunique(),
            "Pallets per Shipment": consolidated_df['Total Pallets'].sum() / len(consolidated_df)
        }

        percentage_change = {
            key: round(((after[key] - before[key]) / before[key]) * 100, 2) for key in before
        }

        comparison_df = pd.DataFrame({"Before": before, "After": after, "% Change": percentage_change})

        comparison_df_dict = comparison_df.to_dict()

        # Create three columns for before, after, and change metrics
        col1, col2, col3 = st.columns(3)

        # Style for metric display
        metric_style = """
                            <div style="
                                background-color: #f0f2f6;
                                padding: 0px;
                                border-radius: 5px;
                                margin: 5px 0;
                            ">
                                <span style="font-weight: bold;">{label}:</span> {value}
                            </div>
                        """

        # Style for percentage changes
        change_style = """
                            <div style="
                                background-color: #e8f0fe;
                                padding: 0px;
                                border-radius: 5px;
                                margin: 5px 0;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            ">
                                <span style="font-weight: bold;">{label}:</span>
                                <span style="color: {color}; font-weight: bold;">{value:+.1f}%</span>
                            </div>
                        """

        # Before consolidation metrics
        with col1:
            st.markdown("##### Before Consolidation")
            st.markdown(metric_style.format(
                label="Days Shipped",
                value=f"{comparison_df_dict['Before']['Days']:,}"
            ), unsafe_allow_html=True)
            st.markdown(metric_style.format(
                label="Pallets Shipped per Day",
                value=f"{comparison_df_dict['Before']['Pallets Per Day']:.1f}"
            ), unsafe_allow_html=True)
            st.markdown(metric_style.format(
                label="Pallets per Shipment",
                value=f"{comparison_df_dict['Before']['Pallets per Shipment']:.1f}"
            ), unsafe_allow_html=True)

        # After consolidation metrics
        with col2:
            st.markdown("##### After Consolidation")
            st.markdown(metric_style.format(
                label="Days Shipped",
                value=f"{comparison_df_dict['After']['Days']:,}"
            ), unsafe_allow_html=True)
            st.markdown(metric_style.format(
                label="Pallets Shipped per Day",
                value=f"{comparison_df_dict['After']['Pallets Per Day']:.1f}"
            ), unsafe_allow_html=True)
            st.markdown(metric_style.format(
                label="Pallets per Shipment",
                value=f"{comparison_df_dict['After']['Pallets per Shipment']:.1f}"
            ), unsafe_allow_html=True)

        # Percentage changes
        with col3:
            st.markdown("##### Percentage Change")
            st.markdown(change_style.format(
                label="Days Shipped",
                value=comparison_df_dict['% Change']['Days'],
                color="blue" if comparison_df_dict['% Change']['Days'] > 0 else "green"
            ), unsafe_allow_html=True)
            st.markdown(change_style.format(
                label="Pallets Shipped per Day",
                value=comparison_df_dict['% Change']['Pallets Per Day'],
                color="green" if comparison_df_dict['% Change']['Pallets Per Day'] > 0 else "red"
            ), unsafe_allow_html=True)
            st.markdown(change_style.format(
                label="Pallets per Shipment",
                value=comparison_df_dict['% Change']['Pallets per Shipment'],
                color="green" if comparison_df_dict['% Change']['Pallets per Shipment'] > 0 else "red"
            ), unsafe_allow_html=True)


        return comparison_df

    def run_agent_query(self,agent, query, dataframe, max_attempts=3):
        """Runs an agent query with up to `max_attempts` retries on failure.

        Args:
            agent: The agent to invoke.
            query (str): The query to pass to the agent.
            dataframe (pd.DataFrame): DataFrame for response context.
            max_attempts (int, optional): Maximum retry attempts. Defaults to 3.

        Returns:
            str: Final answer or error message after attempts.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                # st.info(f"Attempt {attempt + 1} of {max_attempts}...")
                response = agent.invoke(query)
                response_ = agent_wrapper(response, dataframe)

                self.display_agent_steps(response_['steps'])
                # st.success("Query processed successfully.")
                st.write(response_["final_answer"])

                return response_["final_answer"]

            except Exception as e:
                attempt += 1
                # st.warning(f"Error on attempt {attempt}: {e}")

                if attempt == max_attempts:
                    st.error(f"Failed after {max_attempts} attempts. Please revise the query or check the data.")
                    return f"Error: {e}"

    def display_agent_steps(self,steps):
        """Displays agent reasoning steps and associated plots."""
        for i, step in enumerate(steps):
            st.write(step['message'])
            for plot_path in step['plot_paths']:
                display_saved_plot(plot_path)


    def handle_query(self, question):
        """Handles user queries dynamically with conversation history and data processing."""
        chat_history = [{"Human": question}]

        st.info("Extracting parameters from question...")
        extracted_params = get_parameters_values(self.parameters["api_key"], question)
        self.parameters.update(extracted_params)
        chat_history.append({"Agent": f"Parameters extracted: {extracted_params}"})

        # Run cost-saving algorithm
        st.info("Running cost-saving algorithm...")
        self.get_cost_saving_data()

        create_shipment_window_vs_saving_plot(self.parameters['all_results'], self.parameters['best_params'])

        # Identify row with maximum cost savings
        max_savings_row = self.parameters['all_results'].loc[
            self.parameters['all_results']['Cost Savings'].idxmax()
        ].to_dict()
        chat_history.append({"Agent": f"Optimum results: {max_savings_row}"})

        # Agent for cost-saving data analysis
        agent = create_pandas_dataframe_agent(
            self.llm, self.parameters['all_results'],
            verbose=False, allow_dangerous_code=True,
            handle_parsing_errors=True, return_intermediate_steps=True
        )

        st.info("Analyzing the results...")

        with st.expander("**CORRELATION BETWEEN SHIPMENT WINDOW AND KEY METRICS**", expanded=False):
            shipment_query = (
                "Share a quick insights by comparing Shipment Window against Total Shipments, Cost Savings and Total Shipment costs.",
                "The insight should provide overview about how shipment window varies with these factors.",
                "Avoid plots as plot is already there showing the trend, just provide a single or multi-line comment for each comparison.",
                "Use `python_ast_repl_tool` to write a python script and  then print the results in order to pass it to final response.")

            final_answer = self.run_agent_query(agent, shipment_query, self.parameters['all_results'], max_attempts=3)

        chat_history.extend([{"Human": shipment_query}, {"Agent": final_answer}])

        # Determine shipment window
        user_window = None  # Replace with user input logic if needed
        self.parameters["window"] = int(user_window) if user_window else max_savings_row['Shipment Window']

        st.info(f"Consolidating orders for window {self.parameters['window']}...")
        self.consolidate_for_shipment_window()

        # Compare pre- and post-consolidation results
        st.info("Comparing before and after consolidation...")
        comparison_df = self.compare_before_and_after_consolidation()
        comparison_results = comparison_df.to_dict()

        # st.json(comparison_results)
        chat_history.append({"Agent": f"Comparison results: {comparison_results}"})
        create_calendar_heatmap_before_vs_after(self.parameters)

        return chat_history


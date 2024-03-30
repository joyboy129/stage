import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from model import *
import plotly.graph_objects as go

def main():
    st.title("Portfolio Optimization App")
    
    logo = Image.open('logo.png')
    st.sidebar.image(logo, width=100)

    # Features added to the left sidebar
    st.sidebar.header("Features")

    # Data extraction feature
    extract_data_input = st.sidebar.radio("Data Extraction", ("No", "Yes"))
    folder_path = "Data"
    file_path = "Modèle_Belfaa_updated_linearized_1303_simul_noQ.xlsx"
    
    if extract_data_input == "Yes":
        # Perform data extraction
        with st.spinner("Performing data extraction..."):
            extractor = ExcelDataExtractor(file_path, folder_path)
            extractor.extract_data()
        st.success("Data extraction complete!")
    else:
        st.write("Data extraction skipped.")
    

    # Continue with the rest of the main function
    data_processor = DataProcessor(folder_path)
    data_processor.get_data()
    data_processor.get_dict()
    data_processor.month_week_dict()
    data_processor.extract_scenario_charges()
    data_processor.extract_scenario_production()
    data_processor.extract_sim_data()
    data_processor.extract_price_data()
    data_processor.other_data()
    data_processor.compute_tensor()
    
    # Optimization method selection
    optimization_method = st.sidebar.radio("Optimization Method", ("Gurobi","Pulp"))
    if optimization_method == "Pulp":
        with st.spinner("Using Pulp for optimization..."):
            portfolio_model = PortfolioModelPulp(data_processor)
        st.success("Pulp optimization setup complete!")
    elif optimization_method == "Gurobi":
        with st.spinner("Using Gurobi for optimization..."):
            portfolio_model = PortfolioModelGurobi(data_processor)
        st.success("Gurobi optimization setup complete!")
    else:
        st.error("Invalid optimization method. Please choose between 'Pulp' and 'Gurobi'.")
        return
    n = st.sidebar.number_input("Top Scenarios (n)", min_value=1, value=3, step=1)

    # Optimization Button
    optimize_button = st.sidebar.button("Optimize Portfolio")

    if optimize_button:
        with st.spinner("Optimizing portfolio..."):
            portfolio_model.optimize_portfolio()
        st.success("Optimization complete!")

        # Placeholder values for Chiffre d'affaires, Cout de main d'oeuvre, and Charges variables
        CA_placeholder = int(np.round(portfolio_model.CA_expr, 0))
        CMO_placeholder = int(np.round(portfolio_model.CMO_expr, 0))
        CV_placeholder = int(np.round(portfolio_model.CV_expr, 0))
        # Display the placeholder values in a table
        st.table({
            "Chiffre d'affaires": [CA_placeholder],
            "Cout de main d'oeuvre": [CMO_placeholder],
            "Charges variables": [CV_placeholder]
        })

        # Plot the top scenarios
        try:
            with st.spinner("Plotting top scenarios..."):
                simulation_data = portfolio_model.display()
            st.success("Plotting complete!")
            st.dataframe(simulation_data)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Simulation Widget
    simulate_button = st.sidebar.button("Simulate Portfolio")

    if simulate_button:
        with st.spinner("Simulating portfolio..."):
            portfolio_model.get_top_k(n)
            top_k_data = portfolio_model.list_obj
        st.success("Simulation complete!")

        # Create a Plotly figure
        fig = go.Figure()

        # Add trace for the objective values
        fig.add_trace(go.Scatter(x=list(range(1, n+1)), y=top_k_data,
                             mode='markers+lines', marker=dict(color='#2a9d8f', size=10),  # Greenish color
                             hoverinfo='x+y', name='Objective Values'))

        # Set layout for the plot
        fig.update_layout(title="Top " + str(n) + " Scenarios",
                          xaxis_title="Order",
                          yaxis_title="Objective Value",
                          hovermode='closest',
                          showlegend=True,
                          plot_bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                          paper_bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent paper background
                          font=dict(color='#264653', size=14))  # Dark green font color

        # Define a callback function for handling point selection
        def update_selected_point(trace, points, selector):
            if points.point_inds:
                selected_index = points.point_inds[0]
                st.sidebar.write(f"Selected Point Index: {selected_index}")
            # Store the selected index in a variable or perform any other action

        # Attach the callback function to the plot
        fig.data[0].on_click(update_selected_point)
        # Display the plot
        st.plotly_chart(fig)
    
    # Copyright notice
    st.write("\n\n")
    st.write("Copyright © 2024 Les Domaines. All rights reserved.")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()

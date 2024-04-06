import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import os
from utils import *
from model import *
import time
import plotly.graph_objects as go  # Move this import statement to the beginning

def main():
    st.set_page_config(
        page_title="Portfolio Optimization App",
        page_icon=":chart_with_upwards_trend:",
        layout="wide"
    )

    st.title("Portfolio Optimization App")
    
    # Load logo
    logo = Image.open('logo.png')
    st.sidebar.image(logo, width=100)

    # Features added to the left sidebar
    st.sidebar.header("Features")

    # Data extraction feature
    extract_data_input = st.sidebar.radio("Data Extraction", ("No", "Yes"))
    folder_path = "Data"
    file_path = ""

    if extract_data_input == "Yes":
        uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])
        if uploaded_file is not None:
            file_path = os.path.join(folder_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Perform data extraction
            with st.spinner("Performing data extraction..."):
                extractor = ExcelDataExtractor(file_path, folder_path)
                extractor.extract_data()
            st.success("Data extraction complete!")
        else:
            st.write("Please upload an Excel file.")

    else:
        st.write("Data extraction skipped.")
    
    # Continue with the rest of the main function

    premium = st.sidebar.number_input("Premium", min_value=1, value=15, step=1)
    besoin = st.sidebar.number_input("max besoin", min_value=1, value=600, step=1)
    data_processor = DataProcessor(folder_path,premium)
    data_processor.get_assets()

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
        start_time = time.time()  # Start time of optimization
        with st.spinner("Optimizing portfolio..."):
            portfolio_model.optimize_portfolio(besoin)
        st.success("Optimization complete!")
        end_time = time.time()  # End time of optimization
        optimization_time = end_time - start_time  # Calculate optimization time
        st.write(f"Time taken for optimization: {optimization_time:.2f} seconds")  # Display optimization time

        # Placeholder values for Chiffre d'affaires, Cout de main d'oeuvre, and Charges variables
        CA_placeholder = int(np.round(portfolio_model.CA_expr, 0))
        CMO_placeholder = int(np.round(portfolio_model.CMO_expr, 0))
        CV_placeholder = int(np.round(portfolio_model.CV_expr, 0))
        
        data = {
            "Chiffre d'affaires": [CA_placeholder],
            "Charges variables": [CV_placeholder],
            "Resultat":[CA_placeholder-CMO_placeholder-CV_placeholder]
        }

        df = pd.DataFrame(data)
        
        st.table(df)
        main_doeuvre=portfolio_model.maindoeuvre
        fig = go.Figure()

        # Add trace for the objective values
        fig.add_trace(go.Bar(x=list(range(1, 91)), y=list(main_doeuvre),
                             marker=dict(color='#2a9d8f'),  # Greenish color
                             hoverinfo='x+y', name='Main doeuvre'))

        # Set layout for the plot
        fig.update_layout(title='Main doeuvre',
                          xaxis_title="Semaine",
                          yaxis_title="Nombre de main doeuvre",
                          hovermode='closest',
                          showlegend=True,
                          plot_bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                          paper_bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent paper background
                          font=dict(color='#264653', size=14))  # Dark green font color
        st.plotly_chart(fig)
        scenario_dict={}
        for scenario, semaine in zip(portfolio_model.scenario_chosen, portfolio_model.semaines_chosen):
            if scenario not in scenario_dict:
                scenario_dict[scenario] = [semaine]
            else:
                scenario_dict[scenario].append(semaine)

        data = {
            "Scenario index": list(set(portfolio_model.scenario_chosen)),
            "Scenario": [portfolio_model.scenario_couple[i] for i in list(set(portfolio_model.scenario_chosen))],
            "Mois": [portfolio_model.scenario_mois_dict[i] for i in list(set(portfolio_model.scenario_chosen))],
            "Semaines": [list(set(scenario_dict[i]))[0] for i in list(set(portfolio_model.scenario_chosen))],
            "Hectars": [
    np.round(np.dot(
        [1 if i == portfolio_model.scenario_chosen[value] else 0 for value in range(portfolio_model.num_serre)],
        np.array(list(portfolio_model.serre_sau_dict.values()))
    ) ,2)
    for i in list(set(portfolio_model.scenario_chosen))
],
            "Chiffre d'affaire": [portfolio_model.CA_values[j] for j in list(set(portfolio_model.scenario_chosen))],
             "Marge": [portfolio_model.CA_values[j]-portfolio_model.CV_values[j]-portfolio_model.CMO_values[j] for j in list(set(portfolio_model.scenario_chosen))],
             "Taux de marge":[int(100*(portfolio_model.CA_values[j]-portfolio_model.CV_values[j]-portfolio_model.CMO_values[j])/portfolio_model.CA_values[j])
                            for j in list(set(portfolio_model.scenario_chosen))]
        }
        
        df2=pd.DataFrame(data)
        st.table(df2)
        # Plot the top scenarios
        try:
            with st.spinner("Plotting top scenarios..."):
                simulation_data = portfolio_model.display()
            st.success("Plotting complete!")
            st.dataframe(simulation_data)
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Simulation Widget
    simulate_button = st.sidebar.button("Top n scenarios")

    if simulate_button:
        start_time = time.time()  # Start time of simulation
        with st.spinner("Simulating portfolio..."):
            portfolio_model.get_top_k(n,besoin)
            top_k_data = portfolio_model.list_obj
        st.success("Simulation complete!")
        end_time = time.time()  # End time of simulation
        simulation_time = end_time - start_time  # Calculate simulation time
        st.write(f"Time taken for simulation: {simulation_time:.2f} seconds")  # Display simulation time

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
        icons = ["🥇", "🥈", "🥉"]

        # Iterate over the dataframes and display them with rank icons and labels
        for rank, dataframe in enumerate(portfolio_model.dfs[:3]):  # Considering the top 3 for the podium
            st.markdown(f"## {icons[rank]}  Rank {rank + 1}: Chiffre d'affaire est "+str("{:,.0f}".format(np.sum(np.array(dataframe["Chiffre d'affaire"].str.replace(',', '').astype(int))))))
            st.markdown(f'## Marge est: ' +str("{:,.0f}".format(np.sum(np.array(dataframe["Marge"].str.replace(',', '').astype(int))))))
            st.table(dataframe)

        # If there are more than 3 dataframes, display the remaining ones without a rank label
        if len(portfolio_model.dfs) > 3:
            for rank, dataframe in enumerate(portfolio_model.dfs[3:]):
                st.markdown(f"## Rank {rank + 4}")
                st.markdown(f"## Chiffre d'affaire est "+ str("{:,.0f}".format(np.sum(np.array(dataframe["Chiffre d'affaire"].str.replace(',', '').astype(int))))))
                st.markdown(f'## Marge est: ' +str("{:,.0f}".format(np.sum(np.array(dataframe["Marge"].str.replace(',', '').astype(int))))))
                st.table(dataframe)

    # View CSV Files Button
    view_csv_button = st.sidebar.button("View CSV Files")

    if view_csv_button:
        st.sidebar.write("CSV Files:")
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        for csv_file in csv_files:
            csv_data = pd.read_csv(os.path.join(folder_path, csv_file))
            st.write(csv_data)

    st.write("\n\n")
    st.write("Copyright © 2024 Les Domaines Agricoles. All rights reserved.")

if __name__ == "__main__":
    main()
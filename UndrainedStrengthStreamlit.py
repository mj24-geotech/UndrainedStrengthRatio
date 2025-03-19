

#%% Input Libraries
"""
Code for Calculating Undrained Strength Ratios based on Robertson 2020
Author : Mrunmay 
Version 1 : Jan 2024

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mt
import statistics as st
import streamlit as strm

#%% Page 
st.title("Undrained Strength Ratio Analyses : Robertson (2022)")
# User input fields
friction_angle = st.text_input("Control Volume Friction Angle:")
nkt_value = st.text_input("Nkt:")
# File upload option
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
# Display file content if uploaded
if uploaded_file is not None:
    inputdataframe = pd.read_excel(uploaded_file)
    st.write("### Original DataFrame:")
    st.dataframe(inputdataframe)

    # Option to truncate rows
    max_rows = len(inputdataframe)  # Get total number of rows
    truncate_rows = st.slider("Select number of rows to truncate", 0, max_rows, 0)

    if truncate_rows > 0:
        df = inputdataframe.iloc[truncate_rows:]
        st.write("### Truncated DataFrame:")
        st.dataframe(df)
import streamlit as st

st. set_page_config(layout="wide")

st.title("DiaPred: A Deep Learning Based web application for predicting agonists for the GLP-1 Receptor")


st.markdown(
    """DiaPred is a web application dedicated to predicting the potential of molecules to act as GLP-1 receptor agonists. 

    An Interactive Web Tool Utilizing Deep Neural Networks for Predicting GLP-1 Receptor Agonist Activity

    
"""
)
# st.image("path/to/your/image.png", width=300)  # Replace with your image path
from stmol import showmol
import py3Dmol

# 5NX2
# Structure of thermostabilised full-length GLP-1R in complex with a truncated peptide agonist at 3.7 A resolution
xyzview = py3Dmol.view(width=500,height=500, query='pdb:5NX2')

# # Set background color
# xyzview.setBackgroundColor("#0e1117")

# Set cartoon color (optional)
xyzview.setStyle({'cartoon':{'color':'spectrum'}})

# Display molecule
showmol(xyzview, height=500, width=500)


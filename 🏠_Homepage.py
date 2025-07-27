import base64
import os
import streamlit as st
import pandas as pd
import subprocess
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from joblib import load
from scipy.stats import zscore

st.set_page_config(layout="wide")

def desc_calc():
    padel_command = (
        "java -Xms1G -Xmx1G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar "
        "-removesalt -standardizenitro -fingerprints "
        "-descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml "
        "-dir ./ -file descriptors_output.csv"
    )

    try:
        process = subprocess.Popen(padel_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print(f"Error encountered: {error.decode('utf-8')}")
        else:
            print("Descriptor calculation completed successfully.")

        if os.path.exists('molecule.smi'):
            print("'molecule.smi' file exists and was read successfully.")
        else:
            print("'molecule.smi' file does not exist or was not read.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href
  
def build_model(input_data):
    load_model = tf.keras.models.load_model('cnn_classification_model_DiaPred.h5')
    pca = load('pca_model.joblib')
    input_data_pca = pca.transform(input_data)
    input_data_reshaped = np.reshape(input_data_pca, (input_data_pca.shape[0], input_data_pca.shape[1], 1))

    print("Input Data after PCA and Reshaping: ", input_data_reshaped)

    predictions = load_model.predict(input_data_reshaped)
    st.header('**Prediction output**')

    threshold = 0.5
    predicted_classes = (predictions > threshold).astype(int)

    smiles = pd.DataFrame(smiles_input.split('\n'), columns=['SMILES'])

    predicted_labels = np.where(predicted_classes == 1, 'Active', 'Inactive')
    predicted_classes = pd.Series(predicted_labels.flatten(), name='Prediction')
    confidence_score = pd.Series(predictions.flatten(), name='Confidence score')
    df = pd.concat([smiles, predicted_classes, confidence_score], axis=1)

    leverage_threshold = 0.11
    leverage = np.sum(zscore(input_data, axis=0) ** 2, axis=1) / input_data.shape[1]

    applicability_domain = np.where(leverage > leverage_threshold, 'Outside', 'Inside')
    df['Applicability domain'] = applicability_domain

    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

st.title("DiaPred: A Deep Learning Based web application for predicting agonists for the GLP-1 Receptor")


st.markdown(
    """DiaPred is a web application dedicated to predicting the potential of molecules to act as GLP-1 receptor agonists. 

    An Interactive Web Tool Utilizing Deep Neural Networks for Predicting GLP-1 Receptor Agonist Activity

    
"""
)

uploaded_file = st.sidebar.file_uploader("Upload CSV or text file", type=['csv', 'txt'])

if uploaded_file is not None:
    content = uploaded_file.getvalue().decode("utf-8")
    st.sidebar.write("File uploaded successfully:")
    smiles_list = content.splitlines()
    smiles_input = '\n'.join(smiles_list)  
    with open('molecule.smi', 'w') as f:
        f.write('\n'.join(smiles_list))


else:
    example_smiles = "CCC(C)C(C(=O)NC(C)C(=O)NC(CC1=CNC2=CC=CC=C21)C(=O)NC(CC(C)C)C(=O)NC(C(C)C)C(=O)NC(CCCNC(=N)N)C(=O)NCC(=O)NC(CCCNC(=N)N)C(=O)NCC(=O)O)NC(=O)C(CC3=CC=CC=C3)NC(=O)C(CCC(=O)O)NC(=O)C(CCCCNC(=O)COCCOCCNC(=O)COCCOCCNC(=O)CCC(C(=O)O)NC(=O)CCCCCCCCCCCCCCCCC(=O)O)NC(=O)C(C)NC(=O)C(C)NC(=O)C(CCC(=O)N)NC(=O)CNC(=O)C(CCC(=O)O)NC(=O)C(CC(C)C)NC(=O)C(CC4=CC=C(C=C4)O)NC(=O)C(CO)NC(=O)C(CO)NC(=O)C(C(C)C)NC(=O)C(CC(=O)O)NC(=O)C(CO)NC(=O)C(C(C)O)NC(=O)C(CC5=CC=CC=C5)NC(=O)C(C(C)O)NC(=O)CNC(=O)C(CCC(=O)O)NC(=O)C(C)(C)NC(=O)C(CC6=CN=CN6)N"
    smiles_input = st.sidebar.text_area("Paste SMILES data here", example_smiles, height=200)

if st.sidebar.button('Predict'):
    load_data = pd.DataFrame(smiles_input.split('\n'), columns=['SMILES'])
    load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating fingerprints..."):
        output_file_path = "descriptors_output.csv"
        desc_calc()

    st.header('**Calculated molecular fingerprints**')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.write(desc.shape)

    desc_1 = desc.iloc[:, 1:]
    desc_1.reset_index(drop=True, inplace=True)
    num_features = desc_1.shape[1]
    print("Number of features in desc_1: ", num_features)

    build_model(desc_1)

else:
    st.info('Upload or paste SMILES of the compounds')

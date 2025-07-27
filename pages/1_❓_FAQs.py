import streamlit as st

def faqs():
    st.title("Frequently Asked Questions")

    st.header("Question 1: What is GLP-1 Receptor Agonist?")
    st.write("GLP-1 receptor agonists are a class of medications used in the management of type 2 diabetes. They work by mimicking the action of glucagon-like peptide-1 (GLP-1), which stimulates insulin secretion and reduces blood sugar levels.")

    st.header("Question 2: How accurate are the predictions?")
    st.write("The accuracy of the predictions depends on various factors, including the quality of input data, the performance of the machine learning model, and the domain expertise of the developers. While the model strives to provide accurate predictions, it is essential to interpret the results cautiously and consult with healthcare professionals for medical decisions.")

    st.header("Question 3: Can I trust the results?")
    st.write("While the model aims to provide reliable predictions based on the input data and machine learning algorithms, it is not a substitute for professional medical advice. It is essential to consult with healthcare professionals before making any medical decisions based on the results.")

    st.header("Question 4: How can I contribute to improving the model?")
    st.write("Contributions to improving the model, such as providing feedback, reporting issues, or sharing relevant data, are always welcome. Please contact the developers through the provided channels to contribute effectively.")

    st.header("Question 5: Is my data secure?")
    st.write("Protecting user data privacy and security is a top priority. The application adheres to industry-standard security practices to safeguard user data. Data collected through the application is used solely for research and development purposes and is not shared with third parties without user consent.")

    st.header("Question 6: How can I report an issue or provide feedback?")
    st.write("To report an issue or provide feedback, please use the contact form on our website or reach out to us via email. Your input is valuable in improving the application and providing a better user experience.")

# Display FAQs page
faqs()

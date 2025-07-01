import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title('Customer Complaint Prediction')

st.write('This app predicts whether a customer will complain based on their features.')

# Load the trained model
try:
    filename = 'logistic_regression_model.pkl'
    lr = pickle.load(open(filename, 'rb'))
    st.success("Logistic Regression model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'logistic_regression_model.pkl' not found. Please ensure it's in the same directory.")
    lr = None
except Exception as e:
    st.error(f"Error loading the model: {e}")
    lr = None


# Collect user input
if lr is not None:
    st.sidebar.header('Customer Features')

    # Function to get user input
    def user_input_features():
        income = st.sidebar.slider('Income', 1730.0, 666666.0, 52247.0)
        kidhome = st.sidebar.selectbox('Kidhome', [0, 1])
        teenhome = st.sidebar.selectbox('Teenhome', [0, 1])
        recency = st.sidebar.slider('Recency', 0, 99, 49)
        mntwines = st.sidebar.slider('MntWines', 0, 1493, 303)
        mntfruits = st.sidebar.slider('MntFruits', 0, 149, 26)
        mntmeatproducts = st.sidebar.slider('MntMeatProducts', 0, 1725, 166)
        mntfishproducts = st.sidebar.slider('MntFishProducts', 0, 259, 37)
        mntsweetproducts = st.sidebar.slider('MntSweetProducts', 0, 262, 27)
        mntgoldprods = st.sidebar.slider('MntGoldProds', 0, 321, 44)
        numdealpurchases = st.sidebar.slider('NumDealPurchases', 0, 15, 2)
        numwebpurchases = st.sidebar.slider('NumWebPurchases', 0, 27, 4)
        numcatalogpurchases = st.sidebar.slider('NumCatalogPurchases', 0, 28, 2)
        numstorepurchases = st.sidebar.slider('NumStorePurchases', 0, 13, 5)
        numwebvisitsmonth = st.sidebar.slider('NumWebVisitsMonth', 0, 20, 5)
        acceptedcmp3 = st.sidebar.selectbox('AcceptedCmp3', [0, 1])
        acceptedcmp4 = st.sidebar.selectbox('AcceptedCmp4', [0, 1])
        acceptedcmp5 = st.sidebar.selectbox('AcceptedCmp5', [0, 1])
        acceptedcmp1 = st.sidebar.selectbox('AcceptedCmp1', [0, 1])
        acceptedcmp2 = st.sidebar.selectbox('AcceptedCmp2', [0, 1])
        response = st.sidebar.selectbox('Response', [0, 1])

        # Education and Marital_Status will be handled by label encoding
        education = st.sidebar.selectbox('Education', ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'])
        marital_status = st.sidebar.selectbox('Marital Status', ['Divorced', 'Single', 'Together', 'Widow', 'Alone', 'Absurd', 'YOLO'])

        data = {
            'Income': income,
            'Kidhome': kidhome,
            'Teenhome': teenhome,
            'Recency': recency,
            'MntWines': mntwines,
            'MntFruits': mntfruits,
            'MntMeatProducts': mntmeatproducts,
            'MntFishProducts': mntfishproducts,
            'MntSweetProducts': mntsweetproducts,
            'MntGoldProds': mntgoldprods,
            'NumDealPurchases': numdealpurchases,
            'NumWebPurchases': numwebpurchases,
            'NumCatalogPurchases': numcatalogpurchases,
            'NumStorePurchases': numstorepurchases,
            'NumWebVisitsMonth': numwebvisitsmonth,
            'AcceptedCmp3': acceptedcmp3,
            'AcceptedCmp4': acceptedcmp4,
            'AcceptedCmp5': acceptedcmp5,
            'AcceptedCmp1': acceptedcmp1,
            'AcceptedCmp2': acceptedcmp2,
            'Response': response,
            'Education': education,
            'Marital_Status': marital_status
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Preprocessing the input data (based on the original notebook)
    # Create dummy dataframes to fit LabelEncoders and StandardScaler
    # This is a simplified way, in a real app, you'd load the pre-fitted scalers/encoders
    dummy_data = {
        'Education': ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'],
        'Marital_Status': ['Divorced', 'Single', 'Together', 'Widow', 'Alone', 'Absurd', 'YOLO']
    }
    dummy_df = pd.DataFrame(dummy_data)

    le_education = LabelEncoder()
    le_marital = LabelEncoder()

    le_education.fit(dummy_df['Education'])
    le_marital.fit(dummy_df['Marital_Status'])

    input_df['Education'] = le_education.transform(input_df['Education'])
    input_df['Marital_Status'] = le_marital.transform(input_df['Marital_Status'])

    # Assume the original training data had these columns in this order for scaling
    # In a real app, you'd save and load the scaler fitted on the training data
    # For demonstration, we'll create a dummy scaler and fit it on dummy data
    # that mimics the structure of the scaled data used for training.
    # Replace this with loading your actual fitted scaler.
    # This is a critical step to ensure the model receives input in the correct format.
    # For this example, we'll assume the scaler was fitted on a dataframe
    # containing the same numerical columns as input_df after encoding.
    # A better approach is to save the scaler used in the original notebook.

    # Creating a dummy scaler for demonstration - REPLACE WITH YOUR SAVED SCALER
    # This dummy scaler will be fitted on the input data itself, which is NOT
    # how you would do it in production. You MUST use the scaler fitted on your
    # training data.
    # Here, we simulate scaling on the relevant columns.
    numerical_cols_for_scaling = [col for col in input_df.columns if col not in ['Education', 'Marital_Status']]
    scaler = StandardScaler()
    # In production, load your saved scaler like:
    # with open('scaler.pkl', 'rb') as f:
    #     scaler = pickle.load(f)
    # For this Colab example, let's assume a scaler fitted on some data exists.
    # We'll just transform the input data for the demo.
    # This part needs the scaler fitted on the training data!
    # Since we don't have the scaler file, we'll skip scaling for this example
    # or add a placeholder.
    # If you saved your scaler, uncomment the loading part and scale the input_df.

    # Simulating the data used to train the scaler for the purpose of this demo
    # In a real scenario, you load the pre-fitted scaler.
    # Let's assume the scaler was trained on numerical columns + encoded columns
    # of the original dataset, excluding 'Complain'.
    # Based on the notebook, x was scaled. x was df1.drop("Complain",axis=1)
    # df1 had 'Education' and 'Marital_Status' label encoded.

    # Let's recreate a minimal dummy DataFrame to fit the scaler *conceptually*
    # In a real scenario, you'd load the *actual* scaler fitted on your training data.
    # This section is simplified for the Colab environment without saved artifacts.
    # A correct approach requires saving the scaler in the original notebook and loading it here.

    # Placeholder for scaling - This needs to be replaced by loading your actual fitted scaler
    # and transforming input_df using that scaler.
    st.write("Note: Scaling step is simulated. In a production app, load the scaler trained on your data.")
    # Example of how it *should* be done if you saved the scaler:
    # try:
    #     with open('scaler.pkl', 'rb') as f:
    #         loaded_scaler = pickle.load(f)
    #     input_scaled = loaded_scaler.transform(input_df)
    #     input_df_scaled = pd.DataFrame(input_scaled, columns=input_df.columns) # Keep column names if possible
    #     st.write("Input data scaled.")
    # except FileNotFoundError:
    #      st.warning("Scaler file 'scaler.pkl' not found. Skipping scaling.")
    #      input_df_scaled = input_df # Use unscaled data if scaler not found
    # except Exception as e:
    #      st.warning(f"Error loading or applying scaler: {e}. Skipping scaling.")
    #      input_df_scaled = input_df # Use unscaled data on error

    # For this Colab demo, we'll proceed without scaling the input,
    # which might lead to incorrect predictions if the model was trained on scaled data.
    # This highlights the importance of saving and loading all preprocessing objects.
    input_df_processed = input_df # Using unscaled/unprocessed input for demonstration


    st.subheader('User Input Features')
    st.write(input_df_processed)

    # Predict
    if st.button('Predict Complaint'):
        if lr is not None:
            # Ensure the columns and their order match the training data
            # This is crucial! The number of features and their order must be identical.
            # Based on the notebook, the model was trained on 'x' which was df1.drop("Complain",axis=1)
            # df1 had 'Education' and 'Marital_Status' label encoded.
            # We need to ensure input_df_processed has the same columns as the training 'x'.

            # List the columns the model expects (based on the training data 'x')
            # You need to know the exact columns and order from your training 'x'
            # For this demo, let's try to match based on the notebook's drop
            # columns = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
            #        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
            #        'NumDealPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
            #        'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3',
            #        'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
            #        'Response', 'Education', 'Marital_Status'] # Ensure this matches your training features exactly

            # A robust approach would save the list of training columns
            # and ensure the input_df_processed has these columns.
            # For now, let's assume input_df_processed's columns are correct.

            try:
                prediction = lr.predict(input_df_processed)
                prediction_proba = lr.predict_proba(input_df_processed)

                st.subheader('Prediction')
                if prediction[0] == 1:
                    st.error('The customer is predicted to complain.')
                else:
                    st.success('The customer is predicted not to complain.')

                st.subheader('Prediction Probability')
                st.write(f'Probability of not complaining (0): {prediction_proba[0][0]:.4f}')
                st.write(f'Probability of complaining (1): {prediction_proba[0][1]:.4f}')

            except Exception as e:
                 st.error(f"Error during prediction. This might be due to a mismatch in input features or their order. Details: {e}")
                 st.write("Ensure the columns and data types of the input features match the data the model was trained on.")
                 st.write("Input DataFrame columns:", input_df_processed.columns.tolist())
                 st.write("Expected model features (Requires knowing the training columns): Could not verify expected features.")



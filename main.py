import streamlit as st
import pandas as pd
from joblib import load
from PIL import Image

DATA_PATH = 'data.csv'

@st.cache
def load_data(path):
    data = pd.read_csv(path)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

data_load_state = st.text('Loading data...')
df = load_data(DATA_PATH)
data_load_state.text("Done loading data!")

@st.cache
def agg_data(df, mode):
    dat = df.agg([mode])
    return dat

data_agg_state = st.text('Aggregating data...')
dfMin = agg_data(df, 'min')
dfMax = agg_data(df, 'max')
dfMedian = agg_data(df, 'median')
dfMode = agg_data(df, 'mode')
data_agg_state.text("Done aggregating data!")

st.title('Product Backorder')
st.sidebar.title("Features")

quant_parameter_list = ['national_inv',
                'lead_time',
                'in_transit_qty',
                'sales_6_month',
                'pieces_past_due',
                'perf_6_month_avg',
                'local_bo_qty']

qual_parameter_list = ['potential_issue',
                    'deck_risk',
                    'oe_constraint',
                    'ppap_risk',
                    'stop_auto_buy',
                    'rev_stop']

parameter_input_values=[]
values=[]

model_select = st.selectbox(label='Select Classification Model', options=(('AdaBoost', 'Random Forest')))

for parameter in quant_parameter_list:
	values = st.sidebar.slider(label=parameter, key=parameter, value=float(dfMedian[parameter]), min_value=float(dfMin[parameter]), max_value=float(dfMax[parameter]), step=0.1)
	parameter_input_values.append(values)

for parameter in qual_parameter_list:
    ind = dfMode[parameter].iloc[0]
    values = st.sidebar.selectbox(label=parameter, key=parameter, index=int(ind), options=('Yes', 'No'))
    val = 1 if values == 'Yes' else 0
    parameter_input_values.append(val)

parameter_list = quant_parameter_list + qual_parameter_list
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list)
st.write('\n\n')

model = load('AdaBoost.joblib') if model_select == "AdaBoost" else load('RandomForest.joblib')
yes = Image.open('yes.jpg').resize((200, 300))
no = Image.open('no.jpg').resize((200, 300))

if st.button("Will the product be a backorder?"):
    prediction = model.predict(input_variables)
    #pred = 'No' if prediction == 0 else 'Yes'
    #st.text(pred)
    img = no if prediction == 0 else yes
    st.image(img)

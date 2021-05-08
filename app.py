import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, demo  # import your app modules here
import os
import urllib.request

st.set_page_config(
    page_title="Handwashing WHO DL",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Check for ALEXNET MODEL
allModels = list()
for (dirpath, dirnames, filenames) in os.walk('./machine_learning/model'):
    for i in filenames:
        allModels.append(i)
    break

if('alexnet_128.pt' not in allModels):
    url = 'https://storage.googleapis.com/dl-big-project/alexnet_128.pt'
    urllib.request.urlretrieve(url, './machine_learning/model/alexnet_128.pt')

app = MultiApp()

st.write("""
# WHO Hand Washing Classification 🧼🧼🧼

Due to the recent COVID-19 outbreak, handwashing with soap can be one of the defenses against the virus. 
By practising hand hygiene, it can be used to protect us from these diseases, as such the practice of handwashing at
regular intervals should be encouraged and promoted. With the seven-step hand washing technique (broken down into 
12 actions) endorsed by the CDC and World Health Organization (WHO), we would like to promote this 
proper hand washing technique to ensure that the hand washing steps are followed correctly. By using machine learning 
to identify if the hand washing steps are being followed correctly, users can be notified if they have missed out on 
some actions.
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Model", model.app)
app.add_app("Demo", demo.app)
# The main app
app.run()

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from enum import Enum
from array import *
from html_utils import ToHtmlList
import complexities

st.image("https://www.physiotherapyjobscanada.ca/files/pictures/Full_Logo_Black.png")

st.markdown('''<span style="color: limegreen; font-weight: bold; font-size: 2.5em"> 
        Welcome to ConcussionRx!Will this change?
        </span>''', 
        unsafe_allow_html=True
)

st.markdown('''<span style="color: black; font-size: 1.3em"> 
        Canada's first and only AI powered tool for subtyping concussion 
        </span>''', 
        unsafe_allow_html=True
)

st.markdown("---")

link = '[ADVANCE Concussion Clinic](https://www.advanceconcussion.com)'

st.markdown("At " + link + " we believe that concussion is best understood in terms of its complexity rather than independent silos of system specific dysfunction. ConcussionRx was created with the Clinician in mind, to help navigate the challenges of concussion management, which requires sophisticated review of rapdily evolving research, consensus statements, and position papers, as well as specialized knowledge of this complex injury. Using machine learning and clinical validation, ConcussionRx delivers a statistically significant and robust classification for subtyping concussion according to 1 of 5 distinct complexity types.",
   unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: limegreen; font-size: 1.2em'>COMPLEXITY SUBTYPE</h2>",
    unsafe_allow_html=True)

complexity_types = ['Extremely Complex', 'Highly Complex', 'Moderately Complex', 'Mildly Complex', 'Minimally Complex']
complexity_colors = ['#ff0f0d', '#ff5656', '#ff924c', '#ffbe59', '#ffdf59']

complexity_title_style = '''
    height: 3em;
    width: 6em;
    border-radius: 5px;
    color: {0};
    border-color: {0};
    border-width: 2px;
    display: flex;
    border-style: solid;
    font-size: 1.0em;
    justify-content: center;
    align-items: center;
    margin: 1px;
    text-align: center;
'''

complexity_divs = ''

for index in range(5):
    formatted_style = complexity_title_style.format(complexity_colors[index])
    complexity_divs += '<div style="{0}">{1}</div>'.format(formatted_style, complexity_types[index])

st.markdown('<div style="display: flex; justify-content: center;">{0}</div>'.format(complexity_divs),
    unsafe_allow_html=True)

blurb_style = '''
    padding-left: 3em;
    padding-right: 3em;
    padding-bottom: 2em;
    padding-top: 1em;
    width: auto;
    background-color: #d3d3d326;
    border-bottom-right-radius: 10px;
    border-bottom-left-radius: 10px;
    font-size: 0.8em;
'''

blurb_header_style = '''
    padding-left: 1.7em;
    padding-right: 3em;
    padding-top: 1em;
    width: auto;
    color: limegreen;
    background-color: #d3d3d326;
    border-top-right-radius: 10px;
    border-top-left-radius: 10px;
    font-size: 1.4em;
    margin-bottom: 0;
'''

st.write("")

st.markdown("<h2 style=\"{0}\">The Harmonization of AI and Deep Clinical Knowledge in Understanding and Managing Concussion</h2>".format(blurb_header_style), unsafe_allow_html=True)

st.markdown("<div style=\"{0}\">Historically, the complexity of concussion was hindered by the oversimplification of the injury, with recent efforts focused on subtyping according to one system. <br>" 
        "<br>"
        "To address the spectrum of systems that may be affected by concussion, the engagement of an interdisciplinary team is essential. The current concussion landscape tasks general practitioners with the responsibility and burden of quarterbacking concussion care. General practitioners are skilled experts, but even the most experienced clinicians may overlook things. The complexities that surround concussion warrant a management approach that marries clinical expertise with machine learning to manage the challenges of translational care. <br>" 
        "<br>"
#       "The joining of clinical judgement with AI offers an opportunity to improve patient care. Neither AI nor clinical judgement are foolproof, but through the harmonization of both clinical expertise and AI the combinatin of the two combining the joint intelligence of AI and clinician can create a safety net, whereby clinician and AI compensate for what the other may overlook. In harmonizing AI and clinical expertise, we can help make meaningful change in patient management and outcomes. <br>" 
        "Research and clinical evidence supporting multi-modal management of concussion continues to mount, as does the challenege of translating this new and evolving body of knowledge to clinical care. This translational effort requires more than clinical expertise alone, highlighting the opportunity of an AI driven approach. The main strength of deep learning is that it can detect, quantify, and classify even that subtlest of patterns in data, offering an extra set of machine learning eyes. " 
        "By automating and streamlining critical and complex tasks, AI frees up the focus, attention and time that clinicians can put towards patient facing interaction and high level clinical value adding tasks. The value of AI is defined by what it offers the clinician." 
        #"For machine learning to be valuable in healthcare, it must contribute and perform on par or even better than a human expert in that specific area. The value of an AI algorithm is defined by what is offer the clinician. "
        "<br>"
        .format(blurb_style), unsafe_allow_html=True)

''' This is the side bar and algorithm '''



st.sidebar.header('Inventory Results')

def user_input_features():
    clientnumber = st.sidebar.text_input(label='Client Number',value='1872')
    experimentID = st.sidebar.text_input(label='Experiment ID',value='AC-0008'),

    #data = {'Experiment ID':experimentID,
            #'Client Number':clientnumber} 

    #features = pd.DataFrame(data, index=[0])
    return experimentID, clientnumber

df = user_input_features()

#st.write(df)
experimentid = df[0][0]
clientnumber = df[1]
#st.write(experimentid)

dataset = pd.read_csv('ACCdatabase2.csv')
experimentids = dataset['Experiment ID'].values
combined = dataset[['Experiment ID','Headache Hx (0/1)\nno=0\nyes=1','Migrain Hx (None 0; Personal 1; Family 2; Personal and Family 3)','Pain Interference Percentile','Pain Intensity','Physical Function and Mobility Percentile','Anxiety Percentile','Depression Percentile','DHI Total', 'DHI Functional','Visual Motor Speed Composite','Reaction Time Composite Score','Memory Composite (Verbal) Score','Sleep Disturbance Percentile','Ability to Participate in Social Roles Percentile','Cognitive Function Percentile','Fatigue Percentile']]
dfnew1 = combined.loc[combined['Experiment ID'] == experimentid]
dfnew = dfnew1.drop(columns = ['Experiment ID','Headache Hx (0/1)\nno=0\nyes=1','Migrain Hx (None 0; Personal 1; Family 2; Personal and Family 3)'])
st.write(dfnew)
#iris = datasets.load_iris()
training_dataset = pd.read_csv('fulltraining.csv')
Y = training_dataset['cluster'] 
X = training_dataset.drop(columns = ['cluster'])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 50)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X, Y)
import joblib 
model= open("randomforest_Classifier.pkl", "rb")
knn_clf=joblib.load(model)

#y_pred = classifier.predict(X_test)
#classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 50)
#classifier.fit(X_train, Y_train)
#Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(Y_test, y_pred)
#s = np.trace(cm)

#st.write(s)
#st.write(len(y_pred))

pred = knn_clf.predict(dfnew)
#st.write(pred)

#st.write(training_dataset.shape) 
#st.write(Y.shape)
#st.write(X.shape)

#X = iris.data
#Y = iris.target

#clf = RandomForestClassifier()
#clf.fit(X, Y)

#pred = clf.predict(df)
prediction_proba = knn_clf.predict_proba(dfnew)

test1 = np.empty(int(round(prediction_proba[0][0] * 100.)))
test1.fill(0.)

test2 = np.empty(int(round(prediction_proba[0][1] * 100.)))
test2.fill(1.)

test3 = np.empty(int(round(prediction_proba[0][2] * 100.)))
test3.fill(2.)

test4 = np.empty(int(round(prediction_proba[0][3] * 100.)))
test4.fill(3.)

test5 = np.empty(int(round(prediction_proba[0][4] * 100.)))
test5.fill(4.)


#st.write(prediction_proba[0])
#st.write(prediction_proba[0][0] * 100.)

st.write("")

st.markdown('''<span style="color: limegreen; font-size: 1.4em"> 
        ConcussionRx Classification 
        </span>''', 
        unsafe_allow_html=True
)

st.write("According to the information provided, ConcussionRx has classified your patient\'s concussion as:")

if pred == 0:
    complexities.highly_complex.render(st)
if pred == 1:
    complexities.moderately_complex.render(st);
if pred == 2:
    complexities.minimally_complex.render(st);
if pred == 3:
    complexities.mildly_complex.render(st);
if pred == 4:
    complexities.extremely_complex.render(st);


st.markdown('''<span style="color: limegreen; font-size: 1.4em"> 
        Visual Summary 
        </span>''', 
        unsafe_allow_html=True
)

#st.write(df)

figure_style = '''
    font-size: 1.2em;
'''

st.markdown("<div style=\"{0}\">Balance and Movement</div>".format(figure_style), unsafe_allow_html=True)
fig = go.Figure()


figure_height = 300
figure_width = 600
font_size = 16


fig.add_trace(go.Indicator(
    value = dfnew['DHI Total'].values[0],
    title = {'text': "Dizziness", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 15], 'color': 'green'},
            {'range': [15,25], 'color': 'orange'},
            {'range': [25,100], 'color': 'red'}]},
    domain = {'row': 1, 'column': 0}))



fig.add_trace(go.Indicator(
    value = dfnew['Physical Function and Mobility Percentile'].values[0],
    title = {'text': "Physical Mobility", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'red'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'green'}]},
    domain = {'row': 1, 'column': 1}))


fig.update_layout(
    width = figure_width, 
    height = figure_height,
    grid = {'rows': 1, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Reaction time "},
        'mode' : "number+gauge"}]
                         }})

st.plotly_chart(fig)

st.markdown("<div style=\"{0}\">Pain and Sleep</div>".format(figure_style), unsafe_allow_html=True)
fig = go.Figure()


fig.add_trace(go.Indicator(
    value = dfnew['Pain Interference Percentile'].values[0],
    title = {'text': "Pain", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 0}))


fig.add_trace(go.Indicator(
    value = dfnew['Sleep Disturbance Percentile'].values[0],
    title = {'text': "Sleep Disturbance", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 1}))



fig.update_layout(
    width = figure_width, 
    height = figure_height,
    grid = {'rows': 1, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Headache"},
        'mode' : "number+gauge"}]
                         }})



st.plotly_chart(fig)

st.markdown("<div style=\"{0}\">Cognition and Memory</div>".format(figure_style), unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Indicator(
    value = dfnew['Memory Composite (Verbal) Score'].values[0],
    title = {'text': "Verbal Memory", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 15], 'color': 'green'},
            {'range': [15,25], 'color': 'orange'},
            {'range': [25,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 0}))


fig.add_trace(go.Indicator(
    value = dfnew['Cognitive Function Percentile'].values[0],
    title = {'text': "Cognitive Impairment", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 1}))


fig.add_trace(go.Indicator(
    mode = "number",
    number={"font":{"size":30}},
    value = dfnew['Reaction Time Composite Score'].values[0],
    title = {'text': "Reaction Time", 'font': {'size': font_size}},
    domain = {'row': 1, 'column': 0}
))


fig.add_trace(go.Indicator(
    #mode = "number+gauge",
    mode = "number",
    number={"font":{"size":30}},
    #gauge = {'shape': "bullet"},
    value = dfnew['Visual Motor Speed Composite'].values[0],
    #domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "Visual Motor Processing Speed", 'font': {'size': font_size}},
    domain = {'row': 1, 'column': 1}))


fig.update_layout(
    width = figure_width, 
    height = 500,
    grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Speed"},
        'mode' : "number+gauge"}]
                         }})

st.plotly_chart(fig)

st.markdown("<div style=\"{0}\">Mood and Activies of Daily Living</div>".format(figure_style), unsafe_allow_html=True)

fig = go.Figure()



fig.add_trace(go.Indicator(
    value = dfnew['Fatigue Percentile'].values[0],
    title = {'text': "Fatigue", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 0}))



fig.add_trace(go.Indicator(
    value = dfnew['Anxiety Percentile'].values[0],
    title = {'text': "Anxiety", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 1}))


fig.add_trace(go.Indicator(
    value = dfnew['Depression Percentile'].values[0],
    title = {'text': "Depression", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 1, 'column': 0}))


fig.add_trace(go.Indicator(
    value = dfnew['Ability to Participate in Social Roles Percentile'].values[0],
    title = {'text': "Social Involvement", 'font': {'size': font_size}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'red'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'green'}]},
    domain = {'row': 1, 'column': 1}))


fig.update_layout(
    width = figure_width, 
    height = 500,
    grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Reaction Time"},
        'mode' : "number+gauge"}]
                         }})

st.plotly_chart(fig)

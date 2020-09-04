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


st.write("""
# Concussion Sub-type classification
""")

st.sidebar.header('Inventory Results')

def user_input_features():
    clientnumber = st.sidebar.number_input(label='Client Number')
    experimentID = st.sidebar.text_input(label='Experiment ID')
    pain_length = st.sidebar.slider('Pain Interference Percentile',0.0,100.0,50.97)
    movement_width = st.sidebar.slider('Physical Function and Mobility',0.0,100.0, 50.79)
    headache_hx = st.sidebar.slider('Headache Hx', 0.0, 1.0,0.0)
    migraine_hx = st.sidebar.slider('Migraine Hx', 0.0, 2.0,0.0)
    Anxiety = st.sidebar.slider('Anxiety Percentile', 0.0, 100.0,48.6)
    pain_intensity = st.sidebar.slider('Pain Intensity', 0.0,10.0,0.2)
    Depression = st.sidebar.slider('Depression Percentile', 0.0, 100.0,46.06)
    DHI_total = st.sidebar.slider('DHI Total', 0.0,100.0,5.9)
    DHI_functional = st.sidebar.slider('DHI Functional', 0.0, 100.0,2.52)
    visual_motor = st.sidebar.slider('Visual Motor Speed Composite', 0.0,60.0,37.544)
    Reaction_time = st.sidebar.slider('Reaction Time Composite Score', 0.0,5.0,0.68)
    memory = st.sidebar.slider('Memory Composite (Verbal) Score', 0.0, 100.0,89.8)
    sleep = st.sidebar.slider('Sleep Disturbance Percentile', 0.0, 100.0,43.6)
    social_roles = st.sidebar.slider('Ability to Participate in Social Roles Percentile', 0.0, 100.0,51.6)
    cognition = st.sidebar.slider('Cognitive Function Percentile', 0.0, 100.0,32.8)
    fatigue = st.sidebar.slider('Fatigue Percentile', 0.0, 100.0,48.2),

    data = {'Pain Interference Percentile': pain_length,
            'Physical Function and Mobility': movement_width,
            'Pain Intensity': pain_intensity,
            'Headache Hx (0/1)\nno=0\nyes=1': headache_hx,
            'Migrain Hx (None 0; Personal 1; Family 2)': migraine_hx,
	    'Anxiety Percentile': Anxiety,
	    'Depression Percentile': Depression, 
            'DHI Total': DHI_total,
	    'DHI Functional': DHI_functional, 
            'Visual Motor Speed Composite':visual_motor,
            'Reaction Time Composite Score': Reaction_time,
            'Memory Composite (Verbal) Score': memory, 
            'Sleep Disturbance Percentile': sleep, 
            'Ability to Participate in Social Roles Percentile': social_roles,
            'Cognitive Function Percentile': cognition,
            'Fatigue Percentile': fatigue} 

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

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

dfnew = df.drop(columns = ['Headache Hx (0/1)\nno=0\nyes=1','Migrain Hx (None 0; Personal 1; Family 2)'])
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

total = list(test1) + list(test2) + list(test3) + list(test4) + list(test5)
g = sns.distplot(total)
ticks = ['S', 'MS', 'M', 'Mo', 'VS']
g.set(xticks=[0,1,2,3,4])
g.set(xticklabels=ticks)

st.pyplot()

st.write(prediction_proba[0])
st.write(prediction_proba[0][0] * 100.)

st.header('Categorization')
counter = 0 
st.subheader('Your Critical Areas')
if df['Pain Interference Percentile'][0] > 50: 
    st.write('Pain')
    counter = counter + 1 

if df['Physical Function and Mobility'][0] < 51: 
    st.write('Mobility')
    counter = counter + 1 

if df['Anxiety Percentile'][0] > 48 or df['Depression Percentile'][0] > 46: 
    st.write('Mood')
    counter = counter + 1 

if df['DHI Total'][0] > 17: 
    st.write('Dizziness')
    counter = counter + 1 

if df['Visual Motor Speed Composite'][0] < 36: 
    st.write('Visual Motor Speed')
    counter = counter + 1 

if df['Reaction Time Composite Score'][0] > 0.67: 
    st.write('Reaction Time')
    counter = counter + 1 

if df['Sleep Disturbance Percentile'][0] > 43: 
    st.write('Sleep')
    counter = counter + 1 

if df['Cognitive Function Percentile'][0] > 42: 
    st.write('Cognitive')
    counter = counter + 1 

if df['Fatigue Percentile'][0] > 48: 
    st.write('Fatigue')
    counter = counter + 1 

if df['Memory Composite (Verbal) Score'][0] < 80: 
    st.write('Memory')
    counter = counter + 1 
#st.write(iris.target_names)

if pred == 0: 
    #st.write('Patient Classification: Concussion Sub-type 0')
    percentage = counter/10.0 * 100. 
    st.subheader('Patient Classification: Severe Concussion')
    st.write('Your symptoms are a ' + str(percentage) + '% match with the severe concussion symptoms')
    st.write('This concussion type is characterized by having the following high, medium, and low critical areas')
    st.markdown(
    '''<span style="color:red">
    Highly Critical Areas: pain, mobility, mood, cognition, sleep, dizziness, fatigue 
    </span>
    ''',
    unsafe_allow_html=True
)
    st.markdown(
    '''
    <span style="color:blue">
    Medium Critical Areas: Memory,Visual Motor Speed, Reaction Time
    </span>
    ''',
    unsafe_allow_html=True
)
    
if pred == 1: 
    percentage = counter/10.0 * 100.
    st.subheader('Patient Classification: Moderately Severe concussion')
    st.write('Your symptoms are a ' + str(percentage) + '% match with the moderately severe concussion symptoms')
    st.write('This concussion type is characterized by having the following high, medium, and low critical areas')
    st.markdown(
    '''<span style="color:red">
    Highly Critical Areas: pain, mobility, sleep, memory,fatigue, dizziness 
    </span>
    ''',
    unsafe_allow_html=True
)
    st.markdown(
    '''
    <span style="color:blue">
    Medium Critical Areas: visual motor speed, reaction time, cognition, mood 
    </span>
    ''',
    unsafe_allow_html=True
)

    
if pred == 2: 
    percentage = counter/5.0 * 100.
    st.subheader('Patient Classification: Mild concussion')
    st.write('Your symptoms are a ' + str(percentage) + '% match with the mild concussion symptoms')
    st.write('This concussion type is characterized by having the following high, medium, and low critical areas')
    st.markdown(
    '''<span style="color:red">
    Highly Critical Areas: pain
    </span>
    ''',
    unsafe_allow_html=True
)
    st.markdown(
    '''
    <span style="color:blue">
    Medium Critical Areas: mobility, mood, sleep, fatigue 
    </span>
    ''',
    unsafe_allow_html=True
)


if pred == 3: 
    percentage = counter/7.0 * 100.
    st.subheader('Patient Classification: Moderate concussion')
    st.write('Your symptoms are a ' + str(percentage) + '% match with the moderate concussion symptoms')
    st.write('This concussion type is characterized by having the following high, medium, and low critical areas')
    st.markdown(
    '''<span style="color:red">
    Highly Critical Areas: pain, mobility, mood, sleep, fatigue 
    </span>
    ''',
    unsafe_allow_html=True
)
    st.markdown(
    '''
    <span style="color:blue">
    Medium Critical Areas: cognition, dizziness
    </span>
    ''',
    unsafe_allow_html=True
)


if pred == 4: 
    percentage = counter/10.0 * 100.
    st.subheader('Patient Classification: Very Severe concussion')
    st.write('Your symptoms are a ' + str(percentage) + '% match with the moderate concussion symptoms')
    st.write('This concussion type is characterized by having the following high, medium, and low critical areas')
    st.markdown(
    '''<span style="color:red">
    Highly Critical Areas: pain, mobility, mood, cognition, sleep, memory, visual motor speed, reaction time, dizziness,fatigue 
    </span>
    ''',
    unsafe_allow_html=True
)




st.subheader('Based on your critical areas we recommend the following treatment:')

#st.write(df)

#st.subheader('Prediction')
#st.write(iris.target_names[prediction])
#st.write(pred)


st.header('Patient Summary')
#st.write(df)

st.subheader('Visual and Dizziness')
fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+delta",
    value = df['Reaction Time Composite Score'][0],
    domain = {'row': 0, 'column': 0}))

fig.add_trace(go.Indicator(
    mode = "number+gauge+delta",
    gauge = {'shape': "bullet"},
    delta = {'reference': 30},
    value = df['Visual Motor Speed Composite'][0],
    #domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "Visual Motor Speed", 'font': {'size': 12}},domain = {'row': 0, 'column': 1}))

fig.add_trace(go.Indicator(
    value = df['DHI Total'][0],
    delta = {'reference': 0},
    title = {'text': "Dizziness", 'font': {'size': 12}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 15], 'color': 'green'},
            {'range': [15,25], 'color': 'orange'},
            {'range': [25,100], 'color': 'red'}]},
    domain = {'row': 1, 'column': 0}))



fig.add_trace(go.Indicator(
    value = df['Memory Composite (Verbal) Score'][0],
    delta = {'reference': 0},
    title = {'text': "Verbal Memory", 'font': {'size': 12}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 15], 'color': 'green'},
            {'range': [15,25], 'color': 'orange'},
            {'range': [25,100], 'color': 'red'}]},
    domain = {'row': 1, 'column': 1}))


fig.update_layout(
    width = 760, 
    height = 500,
    grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Reaction time "},
        'mode' : "number+delta+gauge",
        'delta' : {'reference': 0.67}}]
                         }})


st.plotly_chart(fig)

st.subheader('Pain and Mobility')
fig = go.Figure()

fig.add_trace(go.Indicator(
    value = df['Pain Interference Percentile'][0],
    delta = {'reference': 0},
    title = {'text': "Pain", 'font': {'size': 12}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 0}))


fig.add_trace(go.Indicator(
    value = df['Physical Function and Mobility'][0],
    delta = {'reference': 0},
    title = {'text': "Physical Mobility", 'font': {'size': 12}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'red'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'green'}]},
    domain = {'row': 0, 'column': 1}))


fig.add_trace(go.Indicator(
    mode = "number+delta",
    value = df['Headache Hx (0/1)\nno=0\nyes=1'][0],
    domain = {'row': 1, 'column': 0}))

fig.add_trace(go.Indicator(
    mode = "number+delta",
    title = {'text': "Migraine"},
    value = df['Migrain Hx (None 0; Personal 1; Family 2)'][0],
    domain = {'row': 1, 'column': 1}))

fig.update_layout(
    width = 700, 
    height = 500,
    grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Headache"},
        'mode' : "number+delta+gauge",
        'delta' : {'reference': 0.0}}]
                         }})

st.plotly_chart(fig)

st.subheader('Fatigue and Sleep')

fig = go.Figure()

fig.add_trace(go.Indicator(
    value = df['Sleep Disturbance Percentile'][0],
    delta = {'reference': 0},
    title = {'text': "Sleep Disturbance", 'font': {'size': 12}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 0}))

fig.add_trace(go.Indicator(
    value = df['Fatigue Percentile'][0],
    delta = {'reference': 0},
    title = {'text': "Fatigue", 'font': {'size': 12}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 1}))

fig.update_layout(
    width = 700, 
    height = 400,
    grid = {'rows': 1, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Speed"},
        'mode' : "number+delta+gauge",
        'delta' : {'reference': 90}}]
                         }})

st.plotly_chart(fig)

st.subheader('Cognitive and Mood')

fig = go.Figure()

fig.add_trace(go.Indicator(
    value = df['Cognitive Function Percentile'][0],
    delta = {'reference': 0},
    title = {'text': "Cognitive Impairment", 'font': {'size': 14}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 0}))

fig.add_trace(go.Indicator(
    value = df['Anxiety Percentile'][0],
    delta = {'reference': 0},
    title = {'text': "Anxiety", 'font': {'size': 14}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 0, 'column': 1}))

fig.add_trace(go.Indicator(
    value = df['Depression Percentile'][0],
    delta = {'reference': 0},
    title = {'text': "Depression", 'font': {'size': 14}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'green'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'red'}]},
    domain = {'row': 1, 'column': 0}))

fig.add_trace(go.Indicator(
    value = df['Ability to Participate in Social Roles Percentile'][0],
    delta = {'reference': 0},
    title = {'text': "Social Life", 'font': {'size': 14}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 40], 'color': 'red'},
            {'range': [40,60], 'color': 'orange'},
            {'range': [60,100], 'color': 'green'}]},
    domain = {'row': 1, 'column': 1}))


fig.update_layout(
    width = 700, 
    height = 600,
    grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Speed"},
        'mode' : "number+delta+gauge",
        'delta' : {'reference': 90}}]
                         }})

st.plotly_chart(fig)

st.header('Patient Progress')




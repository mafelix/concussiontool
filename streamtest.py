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


st.write("""
# Concussion Sub-Type Classification
""")

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
#st.write(dfnew)
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



st.header('Treatment Rx')
st.write('According to your completed inventories and information provided, the ConcussionRx algorithm has classified your concussion as:')
if pred == 0: 
    #st.write('Patient Classification: Concussion Sub-type 0')

    st.markdown("<h1 style='text-align: center; color: black;'>Highly Complex</h1>", unsafe_allow_html=True)
    #st.subheader('Severe Concussion')
    #st.write('to a 96%'  + ' confidence interval.')


    st.write('This type of concussion complexity most often demonstrates with the following critical areas:')
    st.markdown(
    '''<span style="color:red">
    High Critical Areas: Pain, Mobility, Mood, Cognition, Sleep, Dizziness, Fatigue 
    </span>
    ''',
    unsafe_allow_html=True
)
    st.markdown(
    '''
    <span style="color:orange">
    Medium Critical Areas: Memory,Visual Motor Speed, Reaction Time
    </span>
    ''',
    unsafe_allow_html=True
)
    #st.write('These critical areas are some of the aspects that your Advance Concussion Team will assess, treat and recover:')
    link = '[Advanced Concussion Team](https://www.advanceconcussion.com)'
    st.markdown('These critical areas are some of the aspects that your '+ link +' will assess, treat and recover:', unsafe_allow_html=True)


    st.header('For Your Family Doctor: Treatment Recommendations:')
    st.write('Your patient has completed the ConcussionRx, a clinical screening tool that evaluates the complexity of their concussion and accordingly, the required interdisplinary team needed to optimize patient treatment and recovery.')
    st.write('This approach to concussion classification was developed through an artificial intelligence approach. Using robust statistical analyses, a 95% confidence interval and expert clinical and research consensus, your patient requires the following interdisciplinary team:')
    st.write('Physiotherapy,occupational therapy,kinesiology, counseling, neuropsychology')

    st.write('The following assessments from each clinician are required:')

    st.subheader('Physiotherapy:')
    st.write('query cervical dysfunction, query occulormotor dysfunction, query vestibular dysfunction, query autonomic dysfunction')
    st.subheader('Neuropsychology:')
    st.write('query cognitive dysfunction')
    st.subheader('Counseling:')
    st.write('query emotional/mood dysfunction')
    st.subheader('Occupational therapy:')
    st.write('query ADLs function in school, work, home')

    link = '[Advanced Concussion Clinic](https://www.advanceconcussion.com)'
    st.markdown('For more information on engaging an interdisplinary concussion team, please contact the '+ link, unsafe_allow_html=True)



    
if pred == 1: 

    st.markdown("<h1 style='text-align: center; color: black;'>Moderately Complex</h1>", unsafe_allow_html=True)
    #st.subheader('Moderately Severe concussion')
    #st.write('to a 96%'  + ' confidence interval.')

    st.write('This type of concussion complexity most often demonstrates with the following critical areas:')
    st.markdown(
    '''<span style="color:red">
    High Critical Areas: Pain, Mobility, Sleep, Memory,Fatigue, Dizziness 
    </span>
    ''',
    unsafe_allow_html=True
)
    st.markdown(
    '''
    <span style="color:orange">
    Medium Critical Areas: Visual Motor Speed, Reaction Time, Cognition, Mood 
    </span>
    ''',
    unsafe_allow_html=True
)
    link = '[Advanced Concussion Team](https://www.advanceconcussion.com)'
    st.markdown('These critical areas are some of the aspects that your '+ link +' will assess, treat and recover:', unsafe_allow_html=True)

    st.header('For Your Family Doctor: Treatment Recommendations:')
    st.write('Your patient has completed the ConcussionRx, a clinical screening tool that evaluates the complexity of their concussion and accordingly, the required interdisplinary team needed to optimize patient treatment and recovery.')
    st.write('This approach to concussion classification was developed through an artificial intelligence approach. Using robust statistical analyses, a 95% confidence interval and expert clinical and research consensus, your patient requires the following interdisciplinary team:')
    st.write('Physiotherapy,occupational therapy,kinesiology, counseling, neuropsychology')
    st.write('The following assessments from each clinician are required:')
    st.subheader('Physiotherapy:')
    st.write('query cervical dysfunction, query occulormotor dysfunction, query vestibular dysfunction, query autonomic dysfunction')
    st.subheader('Neuropsychology:')
    st.write('query cognitive dysfunction')
    st.subheader('counseling:')
    st.write('query emotional/mood dysfunction')
    st.subheader('Occupational therapy:')
    st.write('query ADLs function in school, work, home')

    
if pred == 2: 
    
    st.markdown("<h1 style='text-align: center; color: black;'>Low Complex</h1>", unsafe_allow_html=True)

    #st.subheader('Mild concussion')
    #st.write('to a 96'  + '%  confidence interval.')
   
    st.write('This type of concussion complexity most often demonstrates with the following critical areas:')
    st.markdown(
    '''<span style="color:red">
    High Critical Areas: Pain
    </span>
    ''',
    unsafe_allow_html=True
)
    st.markdown(
    '''
    <span style="color:blue">
    Medium Critical Areas: Mobility, Mood, Sleep, Fatigue 
    </span>
    ''',
    unsafe_allow_html=True
)
    link = '[Advanced Concussion Team](https://www.advanceconcussion.com)'
    st.markdown('These critical areas are some of the aspects that your '+ link +' will assess, treat and recover:', unsafe_allow_html=True)

    st.header('For Your Family Doctor: Treatment Recommendations:')
    st.write('Your patient has completed the ConcussionRx, a clinical screening tool that evaluates the complexity of their concussion and accordingly, the required interdisplinary team needed to optimize patient treatment and recovery.')
    st.write('This approach to concussion classification was developed through an artificial intelligence approach. Using robust statistical analyses, a 95% confidence interval and expert clinical and research consensus, your patient requires the following interdisciplinary team:')
    st.write('Physiotherapy,occupational therapy,kinesiology, counseling')
    st.write('The following assessments from each clinician are required:')
    st.subheader('Physiotherapy:')
    st.write('query cervical dysfunction, query autonomic dysfunction')
    st.subheader('counseling:')
    st.write('query emotional/mood dysfunction')
    st.subheader('Occupational therapy:')
    st.write('query ADLs function in school, work, home')

if pred == 3: 
    
    st.markdown("<h1 style='text-align: center; color: black;'>Complex</h1>", unsafe_allow_html=True)
    #st.subheader('Moderate concussion')
    #st.write('to a 96'  + '%  confidence interval.')
 
    st.write('This type of concussion complexity most often demonstrates with the following critical areas:')
    st.markdown(
    '''<span style="color:red">
    High Critical Areas: Pain, Mobility, Mood, Sleep, Fatigue 
    </span>
    ''',
    unsafe_allow_html=True
)
    st.markdown(
    '''
    <span style="color:orange">
    Medium Critical Areas: Cognition, Dizziness
    </span>
    ''',
    unsafe_allow_html=True
)
    link = '[Advanced Concussion Team](https://www.advanceconcussion.com)'
    st.markdown('These critical areas are some of the aspects that your '+ link +' will assess, treat and recover:', unsafe_allow_html=True)

    st.header('For Your Family Doctor: Treatment Recommendations:')
    st.write('Your patient has completed the ConcussionRx, a clinical screening tool that evaluates the complexity of their concussion and accordingly, the required interdisplinary team needed to optimize patient treatment and recovery.')
    st.write('This approach to concussion classification was developed through an artificial intelligence approach. Using robust statistical analyses, a 95% confidence interval and expert clinical and research consensus, your patient requires the following interdisciplinary team:')
    st.write('Physiotherapy,occupational therapy,kinesiology, counseling, neuropsychology')
    st.write('The following assessments from each clinician are required:')
    st.subheader('Physiotherapy:')
    st.write('query cervical dysfunction, query vestibular dysfunction, query autonomic dysfunction')
    st.subheader('Neuropsychology:')
    st.write('query cognitive dysfunction')
    st.subheader('counseling:')
    st.write('query emotional/mood dysfunction')
    st.subheader('Occupational therapy:')
    st.write('query ADLs function in school, work, home')

if pred == 4: 
    
    st.markdown("<h1 style='text-align: center; color: black;'>Very Complex</h1>", unsafe_allow_html=True)
    #st.subheader('Very Severe concussion')
    #st.write('to a 96'  + '%  confidence interval.')
   
    st.write('This type of concussion complexity most often demonstrates with the following critical areas:')
    st.markdown(
    '''<span style="color:red">
    High Critical Areas: Pain, Mobility, Mood, Cognition, Sleep, Memory, Visual Motor Speed, Reaction Time, Dizziness,Fatigue 
    </span>
    ''',
    unsafe_allow_html=True
)
    link = '[Advanced Concussion Team](https://www.advanceconcussion.com)'
    st.markdown('These critical areas are some of the aspects that your '+ link +' will assess, treat and recover:', unsafe_allow_html=True)

    st.header('For Your Family Doctor: Treatment Recommendations:')
    st.write('Your patient has completed the ConcussionRx, a clinical screening tool that evaluates the complexity of their concussion and accordingly, the required interdisplinary team needed to optimize patient treatment and recovery.')
    st.write('This approach to concussion classification was developed through an artificial intelligence approach. Using robust statistical analyses, a 95% confidence interval and expert clinical and research consensus, your patient requires the following interdisciplinary team:')
    st.write('Physiotherapy,occupational therapy,kinesiology, counseling, neuropsychology')
    st.write('The following assessments from each clinician are required:')
    st.subheader('Physiotherapy:')
    st.write('query cervical dysfunction, query occulormotor dysfunction, query vestibular dysfunction, query autonomic dysfunction')
    st.subheader('Neuropsychology:')
    st.write('query cognitive dysfunction')
    st.subheader('counseling:')
    st.write('query emotional/mood dysfunction')
    st.subheader('Occupational therapy:')
    st.write('query ADLs function in school, work, home')



#st.write(df)

#st.subheader('Prediction')
#st.write(iris.target_names[prediction])
#st.write(pred)


st.header('Patient Summary')
#st.write(df)

st.subheader('Balance and Movement')
fig = go.Figure()

fig.add_trace(go.Indicator(
    value = dfnew['DHI Total'].values[0],
    title = {'text': "Dizziness", 'font': {'size': 12}},
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
    title = {'text': "Physical Mobility", 'font': {'size': 12}},
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
    width = 700, 
    height = 700,
    grid = {'rows': 1, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Reaction time "},
        'mode' : "number+gauge"}]
                         }})


st.plotly_chart(fig)

st.subheader('Pain and Sleep')
fig = go.Figure()

fig.add_trace(go.Indicator(
    value = dfnew['Pain Interference Percentile'].values[0],
    title = {'text': "Pain", 'font': {'size': 12}},
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
    title = {'text': "Sleep Disturbance", 'font': {'size': 12}},
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
    width = 700, 
    height = 700,
    grid = {'rows': 1, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Headache"},
        'mode' : "number+gauge"}]
                         }})



st.plotly_chart(fig)

st.subheader('Cognitive and Memory')

fig = go.Figure()

fig.add_trace(go.Indicator(
    value = dfnew['Memory Composite (Verbal) Score'].values[0],
    title = {'text': "Verbal Memory", 'font': {'size': 12}},
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
    title = {'text': "Cognitive Impairment", 'font': {'size': 14}},
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
    value = dfnew['Reaction Time Composite Score'].values[0],
    domain = {'row': 1, 'column': 0}))

fig.add_trace(go.Indicator(
    mode = "number+gauge",
    gauge = {'shape': "bullet"},
    value = dfnew['Visual Motor Speed Composite'].values[0],
    #domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "Visual Motor Speed", 'font': {'size': 10}},domain = {'row': 1, 'column': 1}))


fig.update_layout(
    width = 700, 
    height = 700,
    grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Speed"},
        'mode' : "number+gauge"}]
                         }})

st.plotly_chart(fig)

st.subheader('Mood and Activies of Daily Living')

fig = go.Figure()



fig.add_trace(go.Indicator(
    value = dfnew['Fatigue Percentile'].values[0],
    title = {'text': "Fatigue", 'font': {'size': 12}},
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
    title = {'text': "Anxiety", 'font': {'size': 14}},
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
    title = {'text': "Depression", 'font': {'size': 14}},
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
    title = {'text': "Social Life", 'font': {'size': 14}},
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
    width = 700, 
    height = 700,
    grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Speed"},
        'mode' : "number+gauge"}]
                         }})

st.plotly_chart(fig)

st.header('Status and Patient Progress')

outcomes = pd.read_csv('OutcomeMeasures.csv')
dataset_client = outcomes.loc[outcomes['Client Number'] == clientnumber]
s = np.arange(len(dataset_client['Anxiety'].values))
newlist = dataset_client['timestamp'].values
dates = [] 
for item in newlist: 
    dates.append(item.split()[0])

mood = [] 
anxiety = dataset_client['Anxiety'].values 
depression = dataset_client['Depression'].values 

if len(anxiety) == len(depression):
    i = 0 
    while i < len(anxiety): 
        mood.append(max(anxiety[i],depression[i]))
        i = i + 1 
else: 
    print ("not enough data")


sns.set()
fig, ax = plt.subplots(4,1,sharex='col',figsize=(10,10))

ax[0].scatter(s,mood)
ax[0].plot(s,mood)
ax[0].set_title('Client Number ' + clientnumber)
#ax[0].set_title('Mood')
ax[3].set_xticks(s)
ax[3].set_xticklabels(dates)
ax[3].set_xlabel('date')

ax[0].set_ylabel('Mood')

ax[1].scatter(s,dataset_client['Cognitive Function'].values)
ax[1].plot(s,dataset_client['Cognitive Function'].values)
#ax[0].set_title('Mood')
ax[1].set_ylabel('Cognitive Impairment')

ax[2].scatter(s,dataset_client['Pain Interferance'].values)
ax[2].plot(s,dataset_client['Pain Interferance'].values)
#ax[0].set_title('Mood')
ax[2].set_ylabel('Pain')

ax[3].scatter(s,dataset_client['Physical Function'].values)
ax[3].plot(s,dataset_client['Physical Function'].values)
#ax[0].set_title('Mood')
ax[3].set_ylabel('Movement')


st.write(fig)








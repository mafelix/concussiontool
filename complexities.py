from query import Query
from complexity import Complexity
from critical_areas import CritialAreas

# Define your complexities

'''This is for pred = 0 - Highly Complex Category orginal title - updated for Cirelle to "highly complex"
'''
global highly_complex
highly_complex = Complexity('Highly Complex')
highly_complex.setHighAreas([CritialAreas.PAIN, CritialAreas.MOBILITY, CritialAreas.MOOD, CritialAreas.COGNITION, CritialAreas.SLEEP, CritialAreas.DIZZINESS, CritialAreas.FATIGUE])
highly_complex.setMediumAreas([CritialAreas.MEMORY, CritialAreas.VISUAL_MOTOR_SPEED, CritialAreas.REACTION_TIME])

physio_query = Query("Physiotherapy Rx", ["Query Cervical Dysfunction", 'Query Oculomotor Dysfunction', 'Query Vestibular Dysfunction', 'Query Autonomic Dysfunction'])
highly_complex.addQuery(physio_query)

neuropsychology_query = Query("Neuropsychology Rx", ['Query Cognitive Dysfunction'])
highly_complex.addQuery(neuropsychology_query)

counseling_query = Query("Counseling Rx", ['Query Emotional/Mood Dysfunction'])
highly_complex.addQuery(counseling_query)
 
occupational_therapy_query = Query("Occupational Therapy Rx", ['Query ADLs Function in School, Work, Home'])
highly_complex.addQuery(occupational_therapy_query)

'''This is for pred = 1 - Moderately Complex Category - in file changed for Cirelle now called "morderately complex"
'''
global moderately_complex
moderately_complex = Complexity('Moderately Complex')
moderately_complex.setHighAreas([CritialAreas.PAIN, CritialAreas.MOBILITY, CritialAreas.SLEEP, CritialAreas.MEMORY, CritialAreas.FATIGUE, CritialAreas.DIZZINESS])
moderately_complex.setMediumAreas([CritialAreas.VISUAL_MOTOR_SPEED, CritialAreas.REACTION_TIME, CritialAreas.COGNITION, CritialAreas.MOOD])

physio_query = Query("Physiotherapy Rx", ['Query Cervical Dysfunction, Query Oculomotor Dysfunction, Query Vestibular Dysfunction, Query Autonomic Dysfunction'])
moderately_complex.addQuery(physio_query)

neuropsychology_query = Query("Neuropsychology Rx", ['Query Cognitive Dysfunction'])
moderately_complex.addQuery(neuropsychology_query)

counseling_query = Query("Counseling Rx", ['Query Emotional/Mood Dysfunction'])
moderately_complex.addQuery(counseling_query)
 
occupational_therapy_query = Query("Occupational Therapy Rx", ['Query ADLs Function in School, Work, Home'])
moderately_complex.addQuery(occupational_therapy_query)

'''This is for pred = 2 - Minimally Complex Category - changed for Cirelle but originally called "low complex"
'''
global minimally_complex
minimally_complex = Complexity('Minimally Complex')
minimally_complex.setHighAreas([CritialAreas.PAIN])
minimally_complex.setMediumAreas([CritialAreas.MOBILITY, CritialAreas.MOOD, CritialAreas.SLEEP, CritialAreas.FATIGUE])

physio_query = Query("Physiotherapy Rx", ['Query Cervical Dysfunction', 'Query Autonomic Dysfunction'])
minimally_complex.addQuery(physio_query)

counseling_query = Query("Counseling Rx", ['Query Emotional/Mood Dysfunction'])
minimally_complex.addQuery(counseling_query)
 
occupational_therapy_query = Query("Occupational Therapy Rx", ['Query ADLs Function in School, Work, Home'])
minimally_complex.addQuery(occupational_therapy_query)

'''This is for pred = 3 Mildly Complex - changed for Cirelle this was oringally called "complex"
'''
global mildly_complex
mildly_complex = Complexity('Mildly Complex')
mildly_complex.setHighAreas([CritialAreas.PAIN, CritialAreas.MOBILITY, CritialAreas.MOOD, CritialAreas.SLEEP, CritialAreas.FATIGUE])
mildly_complex.setMediumAreas([CritialAreas.COGNITION, CritialAreas.DIZZINESS])

physio_query = Query("Physiotherapy Rx", ['Query Cervical Dysfunction', 'Query Vestibular Dysfunction', 'Query Autonomic Dysfunction'])
mildly_complex.addQuery(physio_query)

europsychology_query = Query("Neuropsychology Rx", ['Query Cognitive Dysfunction'])
mildly_complex.addQuery(neuropsychology_query)

counseling_query = Query("Counseling Rx", ['Query Emotional/Mood Dysfunction'])
mildly_complex.addQuery(counseling_query)
 
occupational_therapy_query = Query("Occupational Therapy Rx", ['QQuery ADLs Function in School, Work, Home'])
mildly_complex.addQuery(occupational_therapy_query)

'''This is for pred=4 Extremely Complex - changed for Cirelle this used to be labelled as very complex''' 
global extremely_complex
extremely_complex = Complexity('Extremely Complex')
extremely_complex.setHighAreas([CritialAreas.PAIN, CritialAreas.MOBILITY, CritialAreas.MOOD, CritialAreas.COGNITION, CritialAreas.SLEEP, CritialAreas.MEMORY, CritialAreas.VISUAL_MOTOR_SPEED, CritialAreas.REACTION_TIME, CritialAreas.DIZZINESS, CritialAreas.FATIGUE])
extremely_complex.setMediumAreas([])

physio_query = Query("Physiotherapy Rx", ['Query Cervical Dysfunction', 'Query Oculomotor Dysfunction', 'Query Vestibular Dysfunction', 'Query Autonomic Dysfunction'])
extremely_complex.addQuery(physio_query)

europsychology_query = Query("Neuropsychology Rx", ['Query Cognitive Dysfunction'])
extremely_complex.addQuery(neuropsychology_query)

counseling_query = Query("Counseling Rx", ['Query Emotional/Mood Dysfunction'])
extremely_complex.addQuery(counseling_query)
 
occupational_therapy_query = Query("Occupational Therapy Rx", ['Query ADLs Function in School, Work, Home'])
extremely_complex.addQuery(occupational_therapy_query)
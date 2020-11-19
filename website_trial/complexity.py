from critical_areas import CritialAreas
from html_utils import ToHtmlList



class Complexity:
    def __init__(self, title):
        self.medium_critical_areas = []
        self.high_critical_areas = []
        # A list of query objects
        self.queries = []
        self.title = title

    def setMediumAreas(self, medium_areas):
        self.medium_critical_areas = medium_areas

    def setHighAreas(self, high_areas):
        self.high_critical_areas = high_areas

    def setTitle(self, title):
        self.title = title

    def addQuery(self, query):
        self.queries.append(query)

    """
        Render writes all the complexity based data to the streamlit object
    """
    def render(self, st):
        title_style = """
            text-align: center;
            color: black;
            font-weight: bold;
            margin: 0;
            padding-bottom: 1em;
            padding-top: 0;
            font-size: 1.5em;
            text-align: center; 
            color: black;
        """
        st.markdown("<div style='{0}'>".format(title_style) + self.title + "</div>", unsafe_allow_html=True)
        st.write('This subtype is associated with the following critical areas:')
        if len(self.high_critical_areas) > 0: 
            st.markdown(
                '''<span style="color:red">
                High Critical Areas 
                </span>
                ''',
                unsafe_allow_html=True
            )
            st.markdown(CritialAreas.toHtmlList(self.high_critical_areas), unsafe_allow_html=True)

        if len(self.medium_critical_areas) > 0: 
            st.markdown(
                '''
                <span style="color:orange">
                Medium Critical Areas
                </span>
                ''',
                unsafe_allow_html=True
            )
            st.markdown(CritialAreas.toHtmlList(self.medium_critical_areas), unsafe_allow_html=True)


        st.markdown('''<span style="color: limegreen; font-size: 1.4em"> 
                Assessment Recommendations 
                </span>''', 
                unsafe_allow_html=True
        )

        linktwo = '[ADVANCE Concussion Clinic](https://www.advanceconcussion.com/contact-us/)'

        st.markdown("The following recommendations reflect the current state of knowledge of concussion management and builds on the principles outlined in the most recent consensus statements. First and foremost, this tool is intended to help inform treatment planning and support partnership among clinicians. To learn more on how you can access an interdisplinary team, please contact " + linktwo, unsafe_allow_html=True)

                            
        for query in self.queries:
            query.render(st)





from html_utils import ToHtmlList

class Query:
    def __init__(self, title, queries):
        self.title = title
        self.queries = queries

    def render(self, st):
        st.markdown('<span style="color: black; font-size: 1.2em">{0}</span>'.format(self.title + ":"),
            unsafe_allow_html=True
            )
        st.markdown(ToHtmlList(self.queries), unsafe_allow_html=True)
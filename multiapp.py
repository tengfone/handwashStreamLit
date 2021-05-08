"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st


class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.selectbox(
            'Navigation',
            self.apps,
            format_func=lambda app: app['title'])

        # Sidebar
        st.sidebar.header("About")
        st.sidebar.markdown("""
        This project is done for SUTD AY2021 *50.039:Theory and Practice of Deep Learning*   
        
        By:
        - [Phang Teng Fone](https://github.com/tengfone)
        - [Loh De Rong](https://github.com/derong97)
        - [Joey Yeo Kailing](https://github.com/joyobo)
        - [Yeh Swee Khim](https://github.com/YehSweeKhim)
        
        Visit [Github](https://github.com/huiwen99/HandWash) for Model Report   
        
        Visit [GitHub](https://github.com/tengfone/) for UI
        """)

        app['function']()

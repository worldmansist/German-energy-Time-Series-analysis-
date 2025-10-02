# в терминале 
# cd streamlit-crash-cours
# conda create -n crashcourse python=3.12
# Чтобы создать файл через терминал New-Item -Path app.py -ItemType File
# streamlit run app.py

import streamlit as st
import pandas as pd
import streamlit as st
from numpy.random import default_rng as rng

st.title('Hello world!')
st.header("This is a header with a divider", divider="gray")


col1, col2 = st.columns(2)
with col1:
    x = st.slider('Choose an x value', 1, 10)
with col2: 
    st.write("The value of :yellow[***x***] is", x) 



df = pd.DataFrame(rng(0).standard_normal((20, 3)), columns=["a", "b", "c"])
st.area_chart(df)
import streamlit as st
import pickle
import numpy as np

# load both of pickle files
pipe = pickle.load(open("pipe.pkl","rb"))
df = pickle.load(open("df.pkl","rb"))

st.title("Laptop Price Predictor")

# now creake selectboxes

company = st.selectbox("companies",df["Company"].unique())

type = st.selectbox("typename",df["TypeName"].unique())

ram = st.selectbox("Ram",df["Ram"].unique())

weight = st.selectbox("Weight",df["Weight"].unique())

touch_screen = st.selectbox("touch_screen",["Yes","No"])

ips = st.selectbox("Ips_display",["Yes","No"])

screen_size = st.slider("screen size in inches 10.0, 13.0,15.6")

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu = st.selectbox("Cpu brand",df["cpu brand"].unique())

hdd = st.selectbox("HDD",df["HDD"].unique())

ssd = st.selectbox("SSD",df["SSD"].unique())

graphic = st.selectbox("Graphic",df["Graphic"].unique())

os = st.selectbox("Operating system",df["OS"].unique())



if st.button("Predict price"):
    if ips == "Yes":
        ips = 1
    else:
        ips = 0

    if touch_screen == "Yes":
        touch_screen = 1
    else:
        touch_screen = 0

    x_res = int(resolution.split("x")[0])
    y_res = int(resolution.split("x")[1])

    ppi = (x_res**2+y_res**2)*0.5/screen_size
    query = np.array([company,type,ram,weight,touch_screen,ips,ppi,cpu,hdd,ssd,graphic,os])

    query = query.reshape(1,12)
    st.title("laptop price of given feature is "+str(int(np.exp(pipe.predict(query)[0]))))


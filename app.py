
# to run streamlit app= 'streamlit run app.py'


import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from pickle import load
    
selected = option_menu(
    menu_title = None,
    options=["Home", "Prediction"],
    icons = ['house', 'emoji-dizzy'],
    menu_icon = 'cast',
    default_index=0,
    orientation='horizontal'
)

df=pd.read_csv('CAR DETAILS.csv')
df2=pd.read_csv('car_d2.csv')

if selected == 'Home':
    st.title('Car Price Prediction')
    st.markdown('“I don’t buy fur coats or jewelry. I have old cars.”')
    st.image('old-car.jpg')
    st.markdown("""---""")
    st.markdown("""---""")


    

    st.title('Car Dataset💎')

    df1 = df.drop('selling_price',axis = 1)
    st.dataframe(df1)

    st.subheader('Shape of Datasets')
    st.dataframe(df.shape)
    
    st.subheader('Car Brands present in dataset')
    st.dataframe(df2['name'].unique())
    st.subheader('28 Manufacturer available in Dataset')

if selected == 'Prediction':
    
    # Loading pretrained models from pickle file
    oe=load(open('models/ordinal_encoder.pkl','rb'))
    slr = load(open('models/standard_scaler.pkl', 'rb'))
    gbr=load(open('models/gbr.pkl','rb'))


    st.title('💎 Car Price Prediction 💎')

    

    with st.form('my_form'):
        name = st.selectbox(label='Name', options=df2.name.unique())
        fuel = st.selectbox(label='Fuel', options=df2.fuel.unique())
        stype = st.selectbox(label='Seller Type', options=df2.seller_type.unique())
        trns = st.selectbox(label='Transmission', options=df2.transmission.unique())
        own = st.selectbox(label='Owner', options=df2.owner.unique())

        yr = st.selectbox(label='Name', options=sorted(df2.year.unique()))
        # yr = st.number_input('Enter model year : ')
        kmd = st.select_slider('Km_driven', options=sorted(df2.km_driven.unique()))
        # kmd = st.number_input('Enter km driven : ')

        btn = st.form_submit_button(label='Predict')

        if btn:
             if name and fuel and stype and trns and own and yr and kmd:
                query_cat = pd.DataFrame({'name':[name], 'fuel':[fuel],'seller_type':[stype],'transmission':[trns],'owner':[own]})
                query_num = pd.DataFrame({'year':[yr], 'km_driven':[kmd]})   

                query_cat = oe.transform(query_cat)
                query_num = slr.transform(query_num)

                query_point = pd.concat([pd.DataFrame(query_cat), pd.DataFrame(query_num)], axis=1)

                price = gbr.predict(query_point)

                st.success(f"The Price is $ {round(price[0],0)}")

             else:
                st.error('Please enter all values')
import streamlit as st
import pickle
import numpy as np

def load_model() :
    with open('cricket_pred.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_Team1= data["le_Team1"]
le_Team2= data["le_Team2"]


def show_predict_page() :
    st.title("Match Price Prediction")


    Team1=(
        'India IND',
        'England ENG',
        'Pakistan PAK',
        'South Africa SA',
        'New Zealand NZ',
        'Australia AUS',
        'West Indies WI',
        'Sri Lanka SL',
        'Bangladesh BAN',
        'Afghanistan AFG',
        'Zimbabwe ZIM',
        'Ireland IRE', 'UAE UAE', 'Namibia NAM',
        'Scotland SCO', 'Nepal NEP', 'Oman OMA',
        'Netherlands NED','Papua New Guinea PNG', 'Hong Kong HK',
        'Canada CAN', 'Jersey JSY',
        'Qatar QAT', 'Uganda UGA', 'Kuwait KUW', 'United States USA',
        'Singapore SIN', 'Malaysia MAS', 'Italy ITA', 'Kenya KEN'
    )


    Team2= (
        'Turkey TUR', 'Thailand THA', 'Eswatini SWA', 'Lesotho LES',
        'Estonia EST', 'Gibraltar GIB', 'Greece GRE', 'Maldives MDV',
        'Seychelles SEY', 'Bulgaria BUL', 'Samoa SAM', 'Serbia SER',
        'Rwanda RWA', 'Panama PAN', 'Hungary HUN', 'Belize BIZ',
        'Mozambique MOZ', 'Bahamas BAH', 'Cyprus CYP', 'Bhutan BHU',
        'Sierra Leone SRL', 'Luxembourg LUX', 'Switzerland CHE',
        'Malta MLT', 'Czech Republic CZE', 'Ghana GHA', 'Sweden SWE',
        'Malawi MAW', 'Argentina ARG', 'Finland FIN', 'France FRA',
        'Norway NOR', 'Vanuatu VAN', 'Romania ROM', 'Portugal POR',
        'Botswana BOT', 'Nigeria NGR', 'Austria AUT', 'Cayman Islands CAY',
        'Belgium BEL', 'Isle of Man IOM', 'Guernsey GSY', 'Denmark DEN',
        'Spain ESP', 'Germany GER', 'Bermuda BRM', 'Tanzania TAN',
        'Saudi Arabia SDA', 'Bahrain BRN', 'Malaysia MAS', 'Italy ITA',
        'Kenya KEN', 'Singapore SIN', 'Uganda UGA', 'United States USA',
        'Kuwait KUW', 'Qatar QAT', 'Jersey JSY', 'Canada CAN',
        'Hong Kong HK', 'Papua New Guinea PNG', 'Netherlands NED',
        'Oman OMA', 'Nepal NEP', 'Scotland SCO', 'Namibia NAM', 'UAE UAE',
        'Ireland IRE', 'Zimbabwe ZIM', 'Afghanistan AFG', 'Bangladesh BAN',
        'Sri Lanka SL', 'West Indies WI', 'Australia AUS',
        'New Zealand NZ', 'South Africa SA', 'Pakistan PAK', 'England ENG',
        'Fiji FIJ', 'Cook Islands COK'
    )

    Team_1 = st.selectbox("1st Team",Team1)
    Team_2 = st.selectbox("2nd Team", Team2)
    Rating_1 = st.slider("Rating of team 1",0,290,0)
    Rating_2 = st.slider("Rating of team 2",0,290,0)
    RatingDiff = st.slider("Diffrence between ratings",0,200,0)

    Button_Names = ['Stand1','Stand2']
    Buttons = st.radio("Select stands",Button_Names)


    ok = st.button("Calculate Price")
    if ok:
        X = np.array([[Team_1,Team_2,Rating_1,Rating_2,RatingDiff]])
        X[:,0] = le_Team1.fit_transform(X[:,0])
        X[:,1] = le_Team2.fit_transform(X[:,1])
        X = X.astype(float)


        Price = regressor.predict(X)

        st.subheader(f"The estimated Price is ${Price}")


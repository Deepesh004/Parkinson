import pickle
import streamlit as st

# Loading the saved model
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# Sidebar for navigation and information display
with st.sidebar:
    st.sidebar.title("Parkinsons Diseases Prediction")
    selected = st.sidebar.selectbox("Go to", ["About Parkinson's", "About Features"])
    st.markdown("# Parkinsons Disease Prediction System")
    if selected == "About Parkinson's":
        st.sidebar.markdown("""
            ## About Parkinson's Disease
            <p style='text-align: justify;'>Parkinson's disease is a progressive disorder that affects the nervous system and the parts of the body controlled by the nerves. Symptoms start slowly. The first symptom may be a barely noticeable tremor in just one hand. Tremors are common, but the disorder also may cause stiffness or slowing of movement.
            In the early stages of Parkinson's disease, your face may show little or no expression. Your arms may not swing when you walk. Your speech may become soft or slurred. Parkinson's disease symptoms worsen as your condition progresses over time.
            Although Parkinson's disease can't be cured, medicines might significantly improve your symptoms. Occasionally, a health care professional may suggest surgery to regulate certain regions of your brain and improve your symptoms.</p>
            """, unsafe_allow_html=True)

    elif selected == "About Features":
        st.sidebar.markdown("""
            ## About Features
            **MDVP (Mean Frequency)**:
            <ul>
                <li>MDVP:Fo (Hz): Mean fundamental frequency (pitch) of the voice signal.</li>
                <li>MDVP:Fhi (Hz): Highest fundamental frequency in the voice signal.</li>
                <li>MDVP:Flo (Hz): Lowest fundamental frequency in the voice signal.</li>
            </ul>
            <p>These features capture information about the pitch of the voice, which can be affected in individuals with Parkinson's disease.</p>

            **Jitter**:
            <ul>
                <li>MDVP:Jitter (%): Percentage of jitter in the voice signal.</li>
                <li>MDVP:Jitter (Abs): Absolute jitter in the voice signal.</li>
                <li>MDVP:RAP: Relative average perturbation in the voice signal.</li>
                <li>MDVP:PPQ: Five-point period perturbation quotient.</li>
                <li>Jitter:DDP: Derivative of the absolute jitter.</li>
            </ul>
            <p>These features measure variations in pitch over time, which can indicate instability in vocal cord vibrations associated with Parkinson's disease.</p>

            **Shimmer**:
            <ul>
                <li>MDVP:Shimmer: Variation in amplitude of consecutive periods of the voice signal.</li>
                <li>MDVP:Shimmer (dB): Amplitude variation in decibels.</li>
                <li>Shimmer:APQ3: Amplitude perturbation quotient for 3 periods.</li>
                <li>Shimmer:APQ5: Amplitude perturbation quotient for 5 periods.</li>
                <li>MDVP:APQ: Absolute shimmer value.</li>
                <li>Shimmer:DDA: Average absolute difference between consecutive points in the amplitude of the waveform.</li>
            </ul>
            <p>These features quantify variations in voice amplitude, which may indicate vocal instability associated with Parkinson's disease.</p>

            **Nonlinear Measures**:
            <ul>
                <li>NHR: Noise-to-harmonics ratio.</li>
                <li>HNR: Harmonics-to-noise ratio.</li>
                <li>RPDE: Recurrence period density entropy.</li>
                <li>DFA: Detrended fluctuation analysis.</li>
                <li>spread1, spread2, D2, PPE: Various nonlinear features derived from nonlinear dynamics analysis.</li>
            </ul>
            <p>These features capture nonlinear properties of the voice signal, which may provide additional insights into voice disorders like Parkinson's disease.</p>
        """, unsafe_allow_html=True)

# Page title
st.title("Parkinson's Disease Prediction using ML")
    
# Grouping features based on their similarities
st.subheader("Acoustic Features")
with st.expander("MDVP (Mean Frequency)"):
    fo = st.slider('MDVP:Fo(Hz)', min_value=0, max_value=300, value=150, step=1)
    fhi = st.slider('MDVP:Fhi(Hz)', min_value=0, max_value=300, value=150, step=1)
    flo = st.slider('MDVP:Flo(Hz)', min_value=0, max_value=300, value=150, step=1)

with st.expander("Jitter"):
    Jitter_percent = st.slider('MDVP:Jitter(%)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    Jitter_Abs = st.slider('MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    RAP = st.slider('MDVP:RAP', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    PPQ = st.slider('MDVP:PPQ', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    DDP = st.slider('Jitter:DDP', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

with st.expander("Shimmer"):
    Shimmer = st.slider('MDVP:Shimmer', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    Shimmer_dB = st.slider('MDVP:Shimmer(dB)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    APQ3 = st.slider('Shimmer:APQ3', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    APQ5 = st.slider('Shimmer:APQ5', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    APQ = st.slider('MDVP:APQ', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    DDA = st.slider('Shimmer:DDA', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

st.subheader("Other Features")
with st.expander("Nonlinear Measures"):
    NHR = st.slider('NHR', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    HNR = st.slider('HNR', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    RPDE = st.slider('RPDE', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    DFA = st.slider('DFA', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

with st.expander("Nonlinear Measures (Continued)"):
    spread1 = st.slider('spread1', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    spread2 = st.slider('spread2', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    D2 = st.slider('D2', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    PPE = st.slider('PPE', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
# Code for Prediction
parkinsons_diagnosis = ''
if st.button("Parkinson's Test Result"):
    parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
    if parkinsons_prediction[0] == 1:
        parkinsons_diagnosis = "The person has Parkinson's disease"
    else:
        parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
st.success(parkinsons_diagnosis)

"""Streamlit Community Cloud entry point for the committed engine model."""
import streamlit as st

from src.inference import load_bundle, predict_observation

st.set_page_config(page_title='Engine Condition | Predictive Maintenance', page_icon='⚙️', layout='wide')
st.caption('HARSHIT RATHI · AIML CAPSTONE PROJECT 3')
st.title('Engine condition assessment')
st.write('Enter six sensor readings to classify an engine observation with the trained random forest.')
st.info('Class 0 and class 1 are dataset labels. Their healthy/faulty meanings are unverified. '
        'This demonstration does not predict time to failure or issue maintenance orders.')


@st.cache_resource
def get_model():
    return load_bundle()


try:
    model, profile, metadata = get_model()
except (OSError, ValueError, KeyError) as exc:
    st.error(f'The model could not be loaded: {exc}')
    st.stop()

with st.form('sensor_readings'):
    st.subheader('Sensor readings')
    st.caption('Defaults are training-set medians. Use the dataset’s measurement scale; '
               'pressure and temperature units were not documented.')
    columns = st.columns(3)
    values = {}
    for i, name in enumerate(profile['features']):
        limits = profile['summary'][name]
        with columns[i % 3]:
            values[name] = st.number_input(
                name, value=round(limits['median'], 2), step=1.0 if name == 'Engine rpm' else 0.1,
                format='%.2f', key=name,
                help=f"Observed training range: {limits['minimum']:.2f} to {limits['maximum']:.2f}. "
                     'These are sample ranges, not verified physical limits.')
    submitted = st.form_submit_button('Assess condition', type='primary')

if submitted:
    try:
        frame, prediction, score, outside = predict_observation(values, model, profile)
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
    else:
        st.subheader('Assessment result')
        left, right = st.columns(2)
        left.metric('Predicted dataset class', f'Class {prediction}')
        right.metric('Class 1 model score', f'{score:.2%}')
        st.caption('The score is an uncalibrated model probability for class 1, not a probability of engine failure. '
                   'The model uses a fixed decision threshold of 0.50; an exact tie is assigned to class 0.')
        if outside:
            st.warning('Outside the observed training range: ' + ', '.join(outside) +
                       '. Treat this result cautiously and verify the readings.')
        st.write('Readings used for this assessment')
        st.dataframe(frame, hide_index=True, width='stretch')
        st.download_button('Download assessment CSV',
                           frame.assign(predicted_class=prediction, class_1_score=score).to_csv(index=False),
                           file_name='engine_condition_assessment.csv', mime='text/csv')

with st.expander('About this model'):
    st.write('Trained on six numeric sensors with a stratified 80/20 split and three-fold cross-validation. '
             'The deployed model is loaded from the GitHub repository; it is not retrained when you enter readings.')
    st.write('Before operational use, confirm label meanings and units, validate on new engines and later periods, '
             'and choose an inspection policy using validation data and agreed error costs.')
    st.link_button('View project on GitHub', 'https://github.com/harshitrathi14/capstone-predictivemaintenance')
st.caption('Inputs are processed for this session and are not written to a server-side prediction log.')

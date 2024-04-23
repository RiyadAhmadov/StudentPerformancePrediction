import pandas as pd
import gradio as gr
import tensorflow as tf

# Load the saved models with specifying the loss function
model_G1 = tf.keras.models.load_model("model_G1.h5", compile=False)
model_G2 = tf.keras.models.load_model("model_G2.h5", compile=False)
model_G3 = tf.keras.models.load_model("model_G3.h5", compile=False)

# Compile the models with the appropriate loss function
model_G1.compile(optimizer='adam', loss='mean_squared_error')
model_G2.compile(optimizer='adam', loss='mean_squared_error')
model_G3.compile(optimizer='adam', loss='mean_squared_error')

# Define a function to make predictions
def predict(school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, 
            reason, guardian, traveltime, studytime, failures, schoolsup, famsup, 
            paid, activities, nursery, higher, internet, romantic, famrel, freetime, 
            goout, Dalc, Walc, health, absences):
    # Prepare input data
    new_data = pd.DataFrame([[school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, 
            reason, guardian, traveltime, studytime, failures, schoolsup, famsup, 
            paid, activities, nursery, higher, internet, romantic, famrel, freetime, 
            goout, Dalc, Walc, health, absences]], columns=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 
                                                             'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 
                                                             'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 
                                                             'Dalc', 'Walc', 'health', 'absences'])
    # Process input data
    new_data[['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']] = new_data[['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']].replace({'yes':1,'no':0})
    new_data['famsize'] = new_data['famsize'].replace({'GT3':1,'LE3':0})
    new_data['address'] = new_data['address'].replace({'R':1,'U':0})
    new_data['school'] = new_data['school'].replace({'GP':1,'MS':0})
    new_data['sex'] = new_data['sex'].replace({'F':0,'M':1})
    new_data['Pstatus'] = new_data['Pstatus'].replace({'A':0,'T':1})
    mjob_mapping = {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4}
    fjob_mapping = {'teacher': 4, 'other': 2, 'services': 3, 'health': 1, 'at_home': 0}
    reason_mapping = {'course': 0, 'other': 1, 'home': 2, 'reputation': 3}
    guardian_mapping = {'mother': 1, 'father': 2, 'other': 0}
    new_data['Mjob'] = new_data['Mjob'].replace(mjob_mapping)
    new_data['Fjob'] = new_data['Fjob'].replace(fjob_mapping)
    new_data['reason'] = new_data['reason'].replace(reason_mapping)
    new_data['guardian'] = new_data['guardian'].replace(guardian_mapping)
    new_data = new_data.astype(int)    
    # Alcohol Consumption Score
    new_data['alc_score'] = new_data['Dalc'] + new_data['Walc']

    # Parental Education Average
    new_data['parent_edu_avg'] = (new_data['Medu'] + new_data['Fedu']) / 2

    # Stress Level
    new_data['stress_level'] = new_data['failures'] + new_data['absences']

    # Socializing Score
    new_data['social_score'] = new_data['goout'] + new_data['freetime']

    # Internet and Higher Education
    new_data['internet_higher'] = (new_data['internet'] == 1) & (new_data['higher'] == 1)

    # Total Study Time
    new_data['total_study_time'] = new_data['studytime'] + new_data['traveltime']

    # Parental Cohabitation Status
    new_data['parent_cohabitation'] = (new_data['Pstatus'] == 1)

    # Parental Occupation Match
    new_data['parent_occ_match'] = (new_data['Mjob'] == new_data['Fjob'])

    # Quality of Family Support
    new_data['fam_support_quality'] = new_data['famrel'] * (new_data['famsup'] == 1)

    # Extra-Curricular Engagement
    new_data['extra_curricular'] = (new_data['activities'] == 1) + (new_data['paid'] == 1)

    # Health and Wellness
    new_data['health_wellness'] = new_data['health'] - new_data['absences']

    # Parental Involvement
    new_data['parent_involvement'] = new_data['reason'] + new_data['guardian']

    new_data = new_data.replace({False: 0, True: 1})
    
    # Use the trained models to predict G1 and G2
    new_G1_pred = model_G1.predict(new_data)
    new_G2_pred = model_G2.predict(new_data)

    # Concatenate input data with predicted G1 and G2
    new_data_concat = pd.concat([new_data, pd.DataFrame(new_G1_pred, columns=['G1_pred']), pd.DataFrame(new_G2_pred, columns=['G2_pred'])], axis=1)

    # Use the trained G3 model to predict G3
    new_G3_pred = model_G3.predict(new_data_concat)

    # Extract predictions
    G1_prediction = new_G1_pred[0][0]  # Assuming new_G1_pred is a scalar
    G2_prediction = new_G2_pred[0][0]  # Assuming new_G2_pred is a scalar
    G3_prediction = new_G3_pred[0][0]  # Assuming new_G3_pred is a scalar

    return G1_prediction, G2_prediction, G3_prediction



# Define the input and output components for Gradio
input_components = [
    gr.Dropdown(choices=["GP", "MS"], label="School"),
    gr.Radio(choices=["F", "M"], label="Sex"),
    gr.Number(label="Age"),
    gr.Radio(choices=["U", "R"], label="Address"),
    gr.Dropdown(choices=["LE3", "GT3"], label="Family Size"),
    gr.Radio(choices=["T", "A"], label="Parent Cohabitation Status"),
    gr.Number(label="Mother's Education"),
    gr.Number(label="Father's Education"),
    gr.Dropdown(choices=["at_home", "health", "other", "services", "teacher"], label="Mother's Job"),
    gr.Dropdown(choices=["at_home", "health", "other", "services", "teacher"], label="Father's Job"),
    gr.Dropdown(choices=["course", "other", "home", "reputation"], label="Reason for Choosing School"),
    gr.Dropdown(choices=["mother", "father", "other"], label="Guardian"),
    gr.Number(label="Travel Time"),
    gr.Number(label="Weekly Study Time"),
    gr.Number(label="Failures"),
    gr.Checkbox(label="Extra Educational Support"),
    gr.Checkbox(label="Family Educational Support"),
    gr.Checkbox(label="Extra Paid Classes"),
    gr.Checkbox(label="Extra-curricular Activities"),
    gr.Checkbox(label="Attended Nursery School"),
    gr.Checkbox(label="Wants to Take Higher Education"),
    gr.Checkbox(label="Has Internet Access"),
    gr.Checkbox(label="In a Romantic Relationship"),
    gr.Number(label="Quality of Family Relationships"),
    gr.Number(label="Free Time After School"),
    gr.Number(label="Going Out with Friends"),
    gr.Number(label="Workday Alcohol Consumption"),
    gr.Number(label="Weekend Alcohol Consumption"),
    gr.Number(label="Health Status"),
    gr.Number(label="Absences")
]

output_components = [
    gr.Textbox(label="Predicted Grade 1"),
    gr.Textbox(label="Predicted Grade 2"),
    gr.Textbox(label="Predicted Grade 3")
]

# Create the Gradio interface
app = gr.Interface(fn=predict, inputs=input_components, outputs=output_components)

# Run the Gradio interface
if __name__ == "__main__":
    app.launch()
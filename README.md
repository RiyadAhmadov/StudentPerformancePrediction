# Student Performance Prediction Model in Tenserflow

![Student Performance Prediction](https://sansone-ac.com/wp-content/uploads/2022/12/Does-Classroom-Temperature-Affect-Students-Performance.jpg)

This project aims to predict the final grades of students based on various socio-economic and personal attributes. The dataset used for this prediction contains information about students' demographics, family background, study habits, and other related factors.

## Dataset Description

The dataset includes the following features:

1. **school**: The school the student attends (`'GP'` for Gabriel Pereira or `'MS'` for Mousinho da Silveira).
2. **sex**: Gender of the student (`'F'` for female or `'M'` for male).
3. **age**: Age of the student.
4. **address**: Type of address of the student (`'U'` for urban or `'R'` for rural).
5. **famsize**: Family size (`'LE3'` for less than or equal to 3 or `'GT3'` for greater than 3).
6. **Pstatus**: Parent's cohabitation status (`'T'` for living together or `'A'` for apart).
7. **Medu**: Mother's education level.
8. **Fedu**: Father's education level.
9. **Mjob**: Mother's job.
10. **Fjob**: Father's job.
11. **reason**: Reason for choosing this school.
12. **guardian**: Student's guardian.
13. **traveltime**: Home to school travel time.
14. **studytime**: Weekly study time.
15. **failures**: Number of past class failures.
16. **schoolsup**: Extra educational support.
17. **famsup**: Family educational support.
18. **paid**: Extra paid classes within the course subject.
19. **activities**: Extra-curricular activities.
20. **nursery**: Attended nursery school.
21. **higher**: Wants to take higher education.
22. **internet**: Internet access at home.
23. **romantic**: In a romantic relationship.
24. **famrel**: Quality of family relationships.
25. **freetime**: Free time after school.
26. **goout**: Going out with friends.
27. **Dalc**: Workday alcohol consumption.
28. **Walc**: Weekend alcohol consumption.
29. **health**: Current health status.
30. **absences**: Number of school absences.

## Dataset link: [Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

## Model Prediction

The project utilizes TensorFlow and Keras to build predictive models based on the provided dataset. Three separate models are trained to predict the students' grades at different periods: G1, G2, and the final grade G3.

## Usage

The `predict` function takes input data about a student and returns the predicted grades for G1, G2, and G3. This function can be integrated into applications or used directly via the provided Gradio interface for real-time predictions.

## Dependencies

- pandas
- gradio
- tensorflow

## How to Run

1. Clone the repository to your local machine.
2. Ensure that all dependencies are installed.
3. Run the Gradio interface by executing the provided script.

## Contributors

- Riyad Ahmadov - riyadehmedov03@gmail.com

## You can look at and practice app at [Student Performance Prediction App](https://huggingface.co/spaces/riyadahmadov/student_performance_models)

# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a logistic regression classifier implemented using Scikit-Learn. 
It is designed to predict whether individuals earn more than $50K per year based on census data.

## Intended Use
This model is intended for use by governmental agencies to understand income distributions and by non-profit
organizations to target aid more effectively.
It is suitable for use in demographic studies where income prediction is relevant 
but should not be used for making individual credit or employment decisions.

## Training Data
The training data is derived from the UCI Machine Learning Repository's Adult dataset. 
The data includes age, work class, education, marital status, 
occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country. 
Preprocessing included one-hot encoding of categorical variables and normalization of continuous variables.

## Evaluation Data
The evaluation dataset was split from the same source as the training data, 
constituting 20% of the total dataset. 
It maintains a similar distribution across all features as the training 
dataset to ensure consistency in model evaluation.

## Metrics
The model was evaluated using precision, recall, and F1 score as key metrics. Performance varied significantly across
different slices of data, reflecting diverse demographic groups:

- Overall Precision: 0.73
- Overall Recall: 0.27
- Overall F1 Score: 0.39

Detailed performance metrics for each demographic group are available in the attached `slice_output.txt`.


## Ethical Considerations
There are significant ethical considerations regarding the fairness and bias of the model predictions
across different racial and gender groups. The model demonstrates variable performance across different demographic 
slices, which could lead to biased outcomes if used in sensitive applications. Use in decision-making processes 
affecting individual livelihoods is not recommended without further bias mitigation.

## Caveats and Recommendations
The model does not perform equally well for all groups, especially for some minority communities, 
meaning it might miss important details about these groups. To make the model better and fairer, 
it is suggested to try out more advanced types of models or changing how we prepare the data before it's used by 
the model. Also, it’s important to regularly check if the model is still doing well over time and update the 
information it learns from as society changes. 
This will help ensure that the model stays relevant and fair for everyone.

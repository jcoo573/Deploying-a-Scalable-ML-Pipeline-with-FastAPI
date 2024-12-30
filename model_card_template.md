# Model Card for Jennifer Cook (jcoo573) 

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model predicts whether an individual's income exceeds $50,000 per year based on demographic and employment attributes. It was built using a `RandomForestClassifier` and trained on the Census Income dataset. 

- **Model Name:** Census Income Prediction Model  
- **Model Version:** 1.0.0 
- **Author:** Jennifer Cook  
- **Date Created:** 12/30/2024

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf


## Intended Use
This model is meant for learning purposes. It demonstrates how to build and deploy a machine learning pipeline using FastAPI. It’s specifically for exploring how income levels vary based on different attributes like education, work class, and more.

## Training Data
The model was trained on the 1994 Census Income dataset sourced from https://archive.ics.uci.edu/dataset/20/census+income. Here's a quick overview:

- **Dataset size:** Approximately 48k rows with 14 columns.
- **Main features:**
  - Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
  - Numeric: Age, Hours-per-week, fnlwgt, education_num, capital-loss, capital-gain .
- **Target label:** `salary`, which is binary (`<=50K` or `>50K`).
- **Filters applied to census to get source data:** "A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))"

### Preprocessing:
- All categorical features were converted into numerical format using one-hot encoding.
- The target label was binarized to ensure it works well with the model.

## Evaluation Data
The evaluation data is a test subset taken from the original dataset. The data was split using an 80/20 train-test split. The test data was processed in the same way as the training data, ensuring consistency.


## Metrics
The model was evaluated on three key metrics: precision, recall, and F1-score. These give an idea of how well the model predicts incomes accurately.


### Overall Performance:
| Metric        | Score  |
|---------------|--------|
| Precision     | 0.7391   |
| Recall        | 0.6384   |
| F1-Score      | 0.6851  |


## Ethical Considerations
- **Bias in Data:** The model might mirror biases in the training data. For example, it could underperform for groups that are underrepresented.
- **Decision-Making:** The predictions should not be the only factor in critical decisions like hiring or lending.
- **Privacy:** Since demographic data is sensitive, it’s essential to handle the data responsibly and protect user privacy.

## Caveats and Recommendations
- **Generalization:** This model may not work well for groups or datasets significantly different from the training data.
- **Monitoring:** Regularly check and retrain the model to keep it relevant and accurate.
- **Cautious Use:** Always consider the model's limitations when interpreting results. It's a helpful tool but not a perfect predictor.

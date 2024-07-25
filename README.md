

## Project Title
**Sarcasm Detection Using RandomForestClassifier**

### Project Overview
This project involves building a machine learning model to detect sarcasm in headlines using the RandomForestClassifier algorithm. The model is trained on a dataset of headlines labeled as sarcastic or non-sarcastic and then tested on a separate dataset to evaluate its performance. The key steps include data preprocessing, vectorization of text data using TF-IDF, training the RandomForestClassifier, and making predictions on new data.

### Requirements
To run this project, you need the following libraries installed in your Python environment:
- numpy
- pandas
- matplotlib
- scikit-learn

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Dataset
The project uses two datasets:
1. **Train_Data.csv** - Contains the headlines and their respective labels (1 for sarcastic, 0 for non-sarcastic).
2. **Test_Data.csv** - Contains headlines for which predictions need to be made.

### Code Explanation

1. **Importing Libraries**
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
   from sklearn.model_selection import train_test_split
   from sklearn.feature_extraction.text import TfidfVectorizer
   ```

2. **Loading and Exploring the Training Data**
   ```python
   data = pd.read_csv('Train_Data.csv')
   data.head()
   data.isnull().sum()
   data['is_sarcastic'].value_counts().plot(kind='bar')
   ```

3. **Preparing Data for Model Training**
   ```python
   x = np.array(data['headline'])
   y = np.array(data['is_sarcastic'])
   
   tf = TfidfVectorizer()
   X = tf.fit_transform(x)
   x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

4. **Training the Model**
   ```python
   model = RandomForestClassifier()
   model.fit(x_train, y_train)
   y_pred = model.predict(x_test)
   
   print(f'f1 score = {f1_score(y_test, y_pred)}')
   print(f'confusion matrix \n = {confusion_matrix(y_test, y_pred)}')
   print(f'\naccuracy = {accuracy_score(y_test,y_pred)}')
   ```

5. **Loading and Preprocessing the Test Data**
   ```python
   test_data = pd.read_csv('Test_Data.csv')
   test_data.info()
   test_data.head()
   test_data.isnull().sum()
   ```

6. **Making Predictions on the Test Data**
   ```python
   output = []
   for i in test_data['headline']:
       data = tf.transform([i]).toarray()
       output.append(model.predict(data)) 
   
   pred = [i[0] for i in output]
   ```

7. **Saving the Predictions to a CSV File**
   ```python
   result = pd.DataFrame({'prediction': pred})
   result.to_csv('submission.csv', index=False)
   ```

### Model Performance
The model achieved the following performance metrics on the test data:
- F1 Score: 0.9277
- Confusion Matrix:
  ```
  [[4413  298]
   [ 301 3841]]
  ```
- Accuracy: 0.9323

### Running the Code
1. Ensure all required libraries are installed.
2. Place the `Train_Data.csv` and `Test_Data.csv` files in the same directory as the script.
3. Run the script using a Python interpreter.

```bash
python sarcasm_detection.py
```

The predictions will be saved in a file named `submission.csv`.



---

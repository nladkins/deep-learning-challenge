# deep-learning-challenge
 
# Deep Learning: Charity Funding Predictor

## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. This project applies machine learning and neural network modeling to a provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

This repository includes a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

## Approach

### Preprocess the data

The first step was to pre process the data. In this case, Pandas and Scikit-Learn’s `StandardScaler()` was used to preprocess the dataset in order to compile, train, and evaluate the neural network model.

The following steps were performed to preprocess the data:

    1. Read in the charity_data.csv to a Pandas DataFrame, and identified the following in the dataset:
        * The variable that are considered the target for the model.
        * What variable that are considered the feature for your model?
    2. Dropped the `EIN` and `NAME` columns.
    3. Determined the number of unique values for each column.
    4. Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then checked if the binning was successful.
    7. Use `pd.get_dummies()` to encode categorical variables

### Compile, Train, and Evaluate the Model

Using `TensorFlow`, I attempted to design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Because the goal was to see if a model could be developed that would exceed a 75% accuracy rating, Layers and units were applied a couple times to compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.  The following items were completed:

    1. Created a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
    2. Created the first hidden layer and chose an appropriate activation function.
    3. Added a second hidden layer with an appropriate activation function.
    4. Created an output layer with an appropriate activation function.
    5. Checked the structure of the model.
    6. Compiled and trained the model.
    7. Evaluated the model using the test data to determine the loss and accuracy.
    9. Saved and exported the results to an HDF5 file, and named it `AlphabetSoupCharity.h5`.

### Optimize the Model

A couple of attempts were made to optimize the model.  The first attempt:

    `# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.`
    `nn = tf.keras.models.Sequential()`

    `# First hidden layer`
    `nn.add(tf.keras.layers.Dense(units=32, activation="relu", input_dim=len(X_train[0])))`

    `# Second hidden layer`
    `nn.add(tf.keras.layers.Dense(units=16, activation="relu"))`

    `# Output layer`
    `nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))`

    `# Check the structure of the model`
    `nn.summary()`

The layer parameters were adjusted (`units=`) to attemt to achieve a predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

* Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
  * Dropping more or fewer columns.
  * Creating more bins for rare occurrences in columns.
  * Increasing or decreasing the number of values for each bin.
* Adding more neurons to a hidden layer.
* Adding more hidden layers.
* Using different activation functions for the hidden layers.
* Adding or reducing the number of epochs to the training regimen.

**NOTE**: You will not lose points if your model does not achieve target performance, as long as you make three attempts at optimizing the model in your jupyter notebook.

1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimzation.ipynb`.
2. Import your dependencies, and read in the `charity_data.csv` to a Pandas DataFrame.
3. Preprocess the dataset like you did in Step 1, taking into account any modifications to optimize the model.
4. Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export your results to an HDF5 file, and name it `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report on the Neural Network Model

For this part of the Challenge, you’ll write a report on the performance of the deep learning model you created for AlphabetSoup.

The report should contain the following:

1. **Overview** of the analysis: Explain the purpose of this analysis.

2. **Results**: Using bulleted lists and images to support your answers, address the following questions.

  * Data Preprocessing
    * What variable(s) are considered the target(s) for your model?
    * What variable(s) are considered to be the features for your model?
    * What variable(s) are neither targets nor features, and should be removed from the input data?
  * Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take to try and increase model performance?

3. **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

- - -

## Rubric

[Unit 21 - Deep Learning Homework Rubric - Charity Funding Predictor](https://docs.google.com/document/d/1SLOROX0lqZwa1ms-iRbHMQr1QSsMT2k0boO9YpFBnHA/edit?usp=sharing)

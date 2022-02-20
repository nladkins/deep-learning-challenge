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

Here was the initial definition of the model which included two hidden layers with 32 neurons in the first layer and 16 neurons in the second layer:

    # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
    nn = tf.keras.models.Sequential()

    # First hidden layer
    nn.add(tf.keras.layers.Dense(units=32, activation="relu", input_dim=len(X_train[0])))

    # Second hidden layer
    nn.add(tf.keras.layers.Dense(units=16, activation="relu"))

    # Output layer
    nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Check the structure of the model
    nn.summary()

The model was compiled and fitted and returned the following result (saved as `AlphabetSoupCharity_Optimization.h5`):

    8575/1 - 1s - loss: 0.5847 - accuracy: 0.7261
    Loss: 0.5566743094163455, Accuracy: 0.726064145565033

### Optimize the Model

A couple of attempts were made to optimize the model.  The layer parameters were adjusted (`units=`) to attemt to achieve a predictive accuracy higher than 75%. 

#### Second Attempt

The second attempt included two hidden layers with 75 neurons in the first layer and 25 neurons in the second layer:

    # Optimize the model in order to achieve a target predictive accuracy higher than 75% by adding more layers.
    nn = tf.keras.models.Sequential()

    # First hidden layer
    nn.add(tf.keras.layers.Dense(units=75, activation="relu", input_dim=len(X_train[0])))

    # Second hidden layer
    nn.add(tf.keras.layers.Dense(units=25, activation="relu"))

    # Output layer
    nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Check the structure of the model
    nn.summary()

Once the model was compiled and fitted, this was the result (saved as `reattempt1.h5`):

    8575/1 - 1s - loss: 0.5929 - accuracy: 0.7233
    Loss: 0.5648308661588775, Accuracy: 0.7232652902603149

There was a slight reduction in accuracy, but very nominal.  Let's further increase the neurons in another attempt.

#### Third Attempt

A third attempt was peformed using the following code which increased the neurons in the first layer to 200 and to 150 neurons in the second layer:

    # Optimize the model in order to achieve a target predictive accuracy higher than 75% by adding more layers.
    nn = tf.keras.models.Sequential()

    # First hidden layer
    nn.add(tf.keras.layers.Dense(units=200, activation="relu", input_dim=len(X_train[0])))

    # Second hidden layer
    nn.add(tf.keras.layers.Dense(units=150, activation="relu"))

    # Output layer
    nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Check the structure of the model
    nn.summary()

Once the model was compiled and fitted, this was the result (saved as `reattempt2.h5`):

    8575/1 - 1s - loss: 0.5783 - accuracy: 0.7254
    Loss: 0.5617579796362896, Accuracy: 0.7253644466400146

This latest attempt is a nominal improvement compared to the last attempt, but still lower than the first attempt (albeit nominal).  All attempts to this point fail to achieve the 75% accuracy goal.  Perhaps the next step is to compare models and tune them to determine the most accurate.

### Kerastuner - tune and compare models

Because neither resulted in a 75% accuracy, `kerastuner` was used to create and compile a new Sequential deep learning model with hyperparameter options with the following features

    * Allowed `kerastuner` to select between `relu` and `tanh` activation functions for each hidden layer.
    * Allowed `kerastuner` to decide from 1 to 30 neurons in the first dense layer.
        * **Note:** To limit the tuner runtime, the *step* argument was increased to 5.
    * Allowed `kerastuner` to decide from 1 to 5 hidden layers and 1 to 30 neurons in each dense layer.
    * Created a **Hyperband** tuner instance using the following parameters:
    * The *objective* is "val_accuracy"
    * *max_epochs* equal to 20
    * *hyperband_iterations* equal to two.
    * Ran the `kerastuner` search for best hyperparameters over 20 epochs.
    * Retrieved the top 3 model hyperparameters from the tuner search and print the values.
    * Retrieved the top 3 models from the tuner search and compare their predictive accuracy against the test dataset.

Result:

    Trial 60 Complete [00h 02m 14s]
    val_accuracy: 0.726064145565033

    Best val_accuracy So Far: 0.727580189704895
    Total elapsed time: 00h 43m 46s
    INFO:tensorflow:Oracle triggered exit

Here is the code used to identify the top three models:

    # Get top 3 model hyperparameters and print the values
    top_hyper = tuner.get_best_hyperparameters(3)
    for param in top_hyper:
        print(param.values)

Result:

    {'activation': 'tanh', 'first_units': 11, 'num_layers': 3, 'units_0': 21, 'units_1': 11, 'units_2': 16, 'units_3': 26, 'units_4': 26, 'tuner/epochs': 20, 'tuner/initial_epoch': 7, 'tuner/bracket': 1, 'tuner/round': 1, 'tuner/trial_id': '34e23cc92e75bf7557a088b903f99241'}
    {'activation': 'tanh', 'first_units': 6, 'num_layers': 1, 'units_0': 16, 'units_1': 26, 'units_2': 1, 'units_3': 11, 'units_4': 11, 'tuner/epochs': 20, 'tuner/initial_epoch': 7, 'tuner/bracket': 1, 'tuner/round': 1, 'tuner/trial_id': '7e06b285acf78b20cb6a9bccb5851a6a'}
    {'activation': 'tanh', 'first_units': 11, 'num_layers': 3, 'units_0': 26, 'units_1': 6, 'units_2': 21, 'units_3': 6, 'units_4': 6, 'tuner/epochs': 20, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}

The top three model parameters were using the "tanh" activation.  All three scores are higher than the previous scores to date.  HOWEVER, the difference is nominal and perhaps could be coincidental.  More importantly, all three of them fall below the 75% accuracy threshold being sought.  

## Conclusion

Three attempts were made to compare the models and actions were performed to compare two models and tune them.  However, no model acheives the target performance.  In other words, no algorithm can be acheived to predict whether or not applicants for funding will be successful using the provided data.
# Importing the required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# SVM Model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# Neural Network Model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# For checking normal distribution of the dataset
from scipy.stats import jarque_bera
from numpy.random import randn

# For checking linearity of the dataset
from statsmodels.stats.stattools import durbin_watson

#for dealing with class imbalance
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense



# Load the dataset
data = pd.read_csv('ai4i2020.csv')
# display dataset
data.head()

# Checking shape of the dataset
print("Number of instances & columns: ", data.shape)


# Examine the columns
print("Columns present in the dataset:")
print(data.columns)

# Examine data types
print("\nData types of different columns in the dataset:")
print(data.dtypes)

# Examine the columns and their data types
print(data.info())

# Create a new column for failure type
data['Failure Type'] = np.where(data['TWF'] == 1, 1,
                                     np.where(data['HDF'] == 1, 2,
                                              np.where(data['PWF'] == 1, 3,
                                                       np.where(data['OSF'] == 1, 4,
                                                                np.where(data['RNF'] == 1, 5, 0)))))

# Drop the individual failure type columns
data.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)


for col in data[['Type','Machine failure']]:
    print(data[col].value_counts()) 
    print("****"*8)

ax = plt.figure(figsize=(15,6))
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Type', data=data)
ax.bar_label(ax.containers[0])
plt.title("Type", fontsize=16,color='Red')
ax =plt.subplot(1,2,2)
ax=data['Type'].value_counts().plot.pie(explode=[0.1, 0.1,0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "Type", fontsize = 16,color='Red');

# Count the number of instances in each class
class_counts = data['Machine failure'].value_counts()

# Print the class counts
print("Class 0 (no failure):", class_counts[0])
print("Class 1 (failure):", class_counts[1])

# Calculate the class balance
class_balance = class_counts[1] / (class_counts[0] + class_counts[1])

print("Class balance: {:.2f}%".format(class_balance * 100))


ax = plt.figure(figsize=(15,6))
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Machine failure', data=data)
ax.bar_label(ax.containers[0])
plt.title("Machine failure", fontsize=16,color='Red')
ax =plt.subplot(1,2,2)
ax=data['Machine failure'].value_counts().plot.pie(explode=[0.1, 0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "Machine failure", fontsize = 16,color='Red');

plt.figure(figsize=(15,7))
sns.scatterplot(data=data, x="Torque [Nm]", y="Rotational speed [rpm]", hue="Machine failure",palette="tab10");

plt.figure(figsize=(15,7))
sns.scatterplot(data=data, x="Torque [Nm]", y="Rotational speed [rpm]", hue="Type",palette="tab10");

plt.figure(figsize=(15,7))
sns.scatterplot(data=data, x="Torque [Nm]", y="Rotational speed [rpm]", hue="Failure Type",palette="tab10");

# Check for missing values
print("\nAny missing values in any of the columns:\n")
print(data.isnull())
print(data.isnull().sum())

# display NaN in each column
data.isna().sum()

#Dropping unnecessary column
data = data.drop(['UDI', 'Product ID'],axis=1)    #these columns are of no use as they are not a deciding factor to our Target

# Examine the columns and their data types
print(data.info())

sns.displot(data=data, x="Air temperature [K]", kde=True, bins = 100,color = "red", facecolor = "yellow",height = 5, aspect = 3.5);

print("Maximum value:",  data['Air temperature [K]'].max())
print("Minimum value:",  data['Air temperature [K]'].min())

sns.displot(data=data, x="Process temperature [K]", kde=True, bins = 100,color = "red", facecolor = "lime",height = 5, aspect = 3.5);

print("Maximum value:",  data['Process temperature [K]'].max())
print("Minimum value:",  data['Process temperature [K]'].min())

sns.displot(data=data, x="Rotational speed [rpm]", kde=True, bins = 100,color = "blue", facecolor = "pink",height = 5, aspect = 3.5);

print("Maximum value:",  data['Rotational speed [rpm]'].max())
print("Minimum value:",  data['Rotational speed [rpm]'].min())

sns.displot(data=data, x="Torque [Nm]", kde=True, bins = 100,color = "orange", facecolor = "purple",height = 5, aspect = 3.5);

print("Maximum value:",  data['Torque [Nm]'].max())
print("Minimum value:",  data['Torque [Nm]'].min())

sns.displot(data=data, x="Tool wear [min]", kde=True, bins = 100,color = "brown", facecolor = "blue",height = 5, aspect = 3.5);

print("Maximum value:",  data['Tool wear [min]'].max())
print("Minimum value:",  data['Tool wear [min]'].min())

# Calculate the IQR (Inter Quartile Range)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
data_no_outliers = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

print("\n\nBox Plots for Original Data\n\n")
# Create box plots for original data
plt.figure(figsize=(12, 8))
plt.suptitle("Box Plots for Original Data")
sns.boxplot(data=data, orient='h')
plt.xticks(rotation=90)
plt.show()

print("\n\nBox Plots for Data without Outliers\n\n")
# Create box plots for data without outliers
plt.figure(figsize=(12, 8))
plt.suptitle("Box Plots for Data with Less Outliers")
sns.boxplot(data=data_no_outliers, orient='h')
plt.xticks(rotation=90)
plt.show()

# Checking shape of the dataset after removing outliers and irrelevant features
print("Number of instances & columns: ", data_no_outliers.shape)

# Examine the columns
print("Columns present in the dataset:")
print(data_no_outliers.columns)

# Select the categorical features to encode
categorical_features = ['Type']

# Perform one-hot encoding using pandas' get_dummies() function
encoded_df = pd.get_dummies(data, columns=categorical_features)

# Display the data with one-hot encoding 
print("Data with One-Hot Encoding:")
# Print the encoded dataframe head
print(encoded_df.head())


# Examine the columns
print("Columns in the dataset after dropping the irrelevant and after performing One-hot encoding:")
encoded_df.columns

# Checking shape of the dataset 
print("Number of instances & columns: ", encoded_df.shape)

# Print the dataframe sample before scaling
print(encoded_df.head(5))

# # Feature scaling 
# scaler = MinMaxScaler()
# cols_to_scale = ['Air temperature [K]','Process temperature [K]',	'Rotational speed [rpm]',	'Torque [Nm]',	'Tool wear [min]']

# encoded_df[cols_to_scale] = scaler.fit_transform(encoded_df[cols_to_scale])

# Print the dataframe sample after scaling
print(encoded_df.head(5))

# Examine the columns and their data types after preprocessing steps
print(encoded_df.info())

# Checking final shape of the dataset
encoded_df.shape


# Feature engineering :

#Calculate the difference between air and process temperature
encoded_df['temp_diff'] = encoded_df['Air temperature [K]'] - encoded_df['Process temperature [K]']

# Calculate the percentage of tool wear
encoded_df['tool_wear_perc'] = (encoded_df['Tool wear [min]'] / encoded_df['Tool wear [min]'].max()) * 100

#Calculate the ratio of torque to rotational speed
# data_preprocessed['torque_to_speed'] = data_preprocessed['Torque [Nm]'] / data_preprocessed['Rotational speed [rpm]']


#Calculate the product of torque and rotational speed (power)
encoded_df['power'] = encoded_df['Torque [Nm]'] * encoded_df['Rotational speed [rpm]']


# Display the new features
print(encoded_df.head())


# Check if a column contains infinity
if np.isinf(encoded_df['power']).any():
    print("The column contains infinity values.")

# Check if a column contains very large values
threshold = 1e9  # Define the threshold for very large values
if (encoded_df['power'] > threshold).any():
    print("The column contains very large values.")


# Examine the columns
print("Columns in the dataset:")
encoded_df.info()

# Checking final shape of the dataset
encoded_df.shape

sns.displot(data=encoded_df, x="temp_diff", kde=True, bins = 100,color = "black", facecolor = "grey",height = 5, aspect = 3.5);

sns.displot(data=encoded_df, x="tool_wear_perc", kde=True, bins = 100,color = "black", facecolor = "grey",height = 5, aspect = 3.5);

sns.displot(data=encoded_df, x="power", kde=True, bins = 100,color = "black", facecolor = "grey",height = 5, aspect = 3.5);

# Compute the correlation matrix
corr_matrix = encoded_df.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Select the features with a high correlation with the target variable
corr_threshold = 0.03
important_features = list(corr_matrix[(corr_matrix['Machine failure']) > corr_threshold].index)

# Drop the target variable from the list of important features
important_features.remove('Machine failure')

# Subset the dataset to include only the important features
selected_data = encoded_df[important_features + ['Machine failure']]

print("Important features given by correlation analysis are: ",important_features)


#Dropping unnecessary column as identified above
encoded_df = encoded_df.drop(['Type_H','Type_L', 'Type_M', 'power', 'Rotational speed [rpm]'],axis=1)    #these columns are of no use as they are not a deciding factor to our Target

#we have to drop this feature as well because this feature needs to have the full view of the dataset, which is not possible at runtime
encoded_df = encoded_df.drop(['tool_wear_perc'],axis=1)  

#Dataset after dropping the unnecessary columns 
encoded_df.info()

# Split the data into features and target
X = encoded_df.drop(['Machine failure', 'Failure Type'], axis=1)
y_failure = encoded_df['Machine failure']
y_type = encoded_df[['Failure Type']]

print(X)
print(y_failure)
print(y_type)

# Split the data into training and testing sets
X_train, X_test, y_train_failure, y_test_failure, y_train_type, y_test_type = train_test_split(X, y_failure, y_type, test_size=0.2, random_state=42)


# Print the shapes of the resulting datasets
print("Training set shape for Machine failure prediction:", X_train.shape, y_train_failure.shape)
print("Testing set shape for Machine failure prediction:", X_test.shape, y_test_failure.shape)
print("Training set shape for Machine Failure Type prediction:", X_train.shape, y_train_type.shape)
print("Testing set shape for Machine Failure Type prediction:", X_test.shape, y_test_type.shape)

# Performing Jarque-Bera test for normality
stats, p = jarque_bera(encoded_df)

if p > 0.05:
    print('The data is Normally distributed.')
else:
    print('The data is not normally distributed.')

# Computing the Durbin-Watson test statistic
dw_statistic = np.apply_along_axis(durbin_watson, 0, X)

# Checking if the data is linear
if np.all(dw_statistic > 1.5) and np.all(dw_statistic < 2.5):
    print("The data is linear.")
else:
    print("The data is non-linear.")

#SVM for machine failure prediction
# Create and train the SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train_failure)

# Make predictions on training and testing data
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
train_accuracy = accuracy_score(y_train_failure, y_train_pred) * 100
test_accuracy = accuracy_score(y_test_failure, y_test_pred) * 100
precision = precision_score(y_test_failure, y_test_pred) * 100
recall = recall_score(y_test_failure, y_test_pred) * 100
f1 = f1_score(y_test_failure, y_test_pred) * 100

# Print the results
print("Training Accuracy of SVM:", train_accuracy, "%")
print("Testing Accuracy of SVM:", test_accuracy, "%")
print("Precision of SVM:", precision, "%")
print("Recall of SVM:", recall, "%")
print("F1 Score of SVM:", f1, "%")


# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train_failure)

# Make predictions on training and testing data
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
train_accuracy = accuracy_score(y_train_failure, y_train_pred) * 100
test_accuracy = accuracy_score(y_test_failure, y_test_pred) * 100
precision = precision_score(y_test_failure, y_test_pred) * 100
recall = recall_score(y_test_failure, y_test_pred) * 100
f1 = f1_score(y_test_failure, y_test_pred) * 100

# Print the results
print("Training Accuracy:", train_accuracy, "%")
print("Testing Accuracy:", test_accuracy, "%")
print("Precision:", precision, "%")
print("Recall:", recall, "%")
print("F1 Score:", f1, "%")


# # Logistic regression for machine failure prediction


# Create the Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)

# Define the hyperparameters for grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(logreg_model, param_grid, cv=5)
grid_search.fit(X_train, y_train_failure)

# Get the best model with the optimized hyperparameters
best_logreg_model = grid_search.best_estimator_

# Make predictions on training and testing data
y_train_pred = best_logreg_model.predict(X_train)
y_test_pred = best_logreg_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
train_accuracy = accuracy_score(y_train_failure, y_train_pred) * 100
test_accuracy = accuracy_score(y_test_failure, y_test_pred) * 100
precision = precision_score(y_test_failure, y_test_pred) * 100
recall = recall_score(y_test_failure, y_test_pred) * 100
f1 = f1_score(y_test_failure, y_test_pred) * 100

# Print the results
print("Training Accuracy:", train_accuracy, "%")
print("Testing Accuracy:", test_accuracy, "%")
print("Precision:", precision, "%")
print("Recall:", recall, "%")
print("F1 Score:", f1, "%")




# # Logistic regression for machine failure prediction

# Apply SMOTE to balance the training data for machine failure prediction
sm = SMOTE(random_state=42)
X_train_resampled, y_train_failure_resampled  = sm.fit_resample(X_train, y_train_failure)
X_test_resampled, y_test_failure_resampled = sm.fit_resample( X_test, y_test_failure)
# Create the Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)

# Define the hyperparameters for grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(logreg_model, param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_failure_resampled)

# Get the best model with the optimized hyperparameters
best_logreg_model = grid_search.best_estimator_

# Make predictions on training and testing data
y_train_pred = best_logreg_model.predict(X_train_resampled)
y_test_pred = best_logreg_model.predict(X_test_resampled)

# Calculate accuracy, precision, recall, and F1 score
train_accuracy = accuracy_score(y_train_failure_resampled, y_train_pred) * 100
test_accuracy = accuracy_score(y_test_failure_resampled, y_test_pred) * 100
precision = precision_score(y_test_failure_resampled, y_test_pred) * 100
recall = recall_score(y_test_failure_resampled, y_test_pred) * 100
f1 = f1_score(y_test_failure_resampled, y_test_pred) * 100

# Print the results
print("Training Accuracy:", train_accuracy, "%")
print("Testing Accuracy:", test_accuracy, "%")
print("Precision:", precision, "%")
print("Recall:", recall, "%")
print("F1 Score:", f1, "%")




# # Logistic regression for machine failure prediction

class_weight = {0: 1, 1: 9661/339}  # Class 0 weight = 1, Class 1 weight = ratio

# Create the Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000, class_weight=class_weight)

# Define the hyperparameters for grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(logreg_model, param_grid, cv=5)
grid_search.fit(X_train, y_train_failure)

# Get the best model with the optimized hyperparameters
best_logreg_model = grid_search.best_estimator_

# Make predictions on training and testing data
y_train_pred = best_logreg_model.predict(X_train)
y_test_pred = best_logreg_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
train_accuracy = accuracy_score(y_train_failure, y_train_pred) * 100
test_accuracy = accuracy_score(y_test_failure, y_test_pred) * 100
precision = precision_score(y_test_failure, y_test_pred) * 100
recall = recall_score(y_test_failure, y_test_pred) * 100
f1 = f1_score(y_test_failure, y_test_pred) * 100

# Print the results
print("Training Accuracy:", train_accuracy, "%")
print("Testing Accuracy:", test_accuracy, "%")
print("Precision:", precision, "%")
print("Recall:", recall, "%")
print("F1 Score:", f1, "%")




model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_train_failure)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_failure, epochs=100, batch_size=32, validation_data=(X_test, y_test_failure))


# Evaluate the model on training data
train_loss, train_accuracy = model.evaluate(X_train, y_train_failure, verbose=0)
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))

# Evaluate the model on testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test_failure, verbose=0)
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

# Predict on testing data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test_failure, y_pred_classes, average='macro')
recall = recall_score(y_test_failure, y_pred_classes, average='macro')
f1 = f1_score(y_test_failure, y_pred_classes, average='macro')

print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))



# Create the Random Forest model with appropriate hyperparameters
model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train_type)

# Predict on the training data
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train_type, y_train_pred)
train_precision = precision_score(y_train_type, y_train_pred, average='macro')
train_recall = recall_score(y_train_type, y_train_pred, average='macro')
train_f1 = f1_score(y_train_type, y_train_pred, average='macro')

# Predict on the testing data
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test_type, y_test_pred)
test_precision = precision_score(y_test_type, y_test_pred, average='macro')
test_recall = recall_score(y_test_type, y_test_pred, average='macro')
test_f1 = f1_score(y_test_type, y_test_pred, average='macro')

# Print the evaluation metrics
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))
print("Precision: {:.2f}%".format(test_precision * 100))
print("Recall: {:.2f}%".format(test_recall * 100))
print("F1 Score: {:.2f}%".format(test_f1 * 100))


# Define the hyperparameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_layers': [1, 2, 3],
    'neurons_per_layer': [64, 128, 256],
    'activation': ['relu', 'sigmoid']
}

# Define the model building function
def build_model(learning_rate, hidden_layers, neurons_per_layer, activation):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation=activation, input_shape=(X_train.shape[1],)))
    
    for _ in range(hidden_layers-1):
        model.add(Dense(neurons_per_layer, activation=activation))
    
    model.add(Dense(len(np.unique(y_train_type)), activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create the model
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train_type)

# Print the best hyperparameter combination and the corresponding validation score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Validation Score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test_type)
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

# Evaluate the best model on the training set
train_accuracy = best_model.score(X_train, y_train_type)
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate precision, recall, and F1 score on the test set
precision = precision_score(y_test_type, y_pred, average='macro')
recall = recall_score(y_test_type, y_pred, average='macro')
f1 = f1_score(y_test_type, y_pred, average='macro')

print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))



model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_train_type)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_type, epochs=100, batch_size=32, validation_data=(X_test, y_test_type))


# Evaluate the model on training data
train_loss, train_accuracy = model.evaluate(X_train, y_train_type, verbose=0)
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))

# Evaluate the model on testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test_type, verbose=0)
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

# Predict on testing data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test_type, y_pred_classes, average='macro')
recall = recall_score(y_test_type, y_pred_classes, average='macro')
f1 = f1_score(y_test_type, y_pred_classes, average='macro')

print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))


# Create the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(len(np.unique(y_train_type)), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Apply SMOTE to balance the training data for machine failure prediction
sm = SMOTE(random_state=42)
X_train_resampled, y_train_type_resampled  = sm.fit_resample(X_train, y_train_type)
X_test_resampled, y_test_type_resampled = sm.fit_resample( X_test, y_test_type)


# Train the model
history = model.fit(X_train_resampled, y_train_type_resampled, epochs=100, batch_size=32, validation_data=(X_test_resampled, y_test_type_resampled))

# Evaluate the model on training data
train_loss, train_accuracy = model.evaluate(X_train_resampled, y_train_type_resampled, verbose=0)
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))

# Evaluate the model on testing data
test_loss, test_accuracy = model.evaluate(X_test_resampled, y_test_type_resampled, verbose=0)
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

# Predict on testing data
y_pred = model.predict(X_test_resampled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test_type_resampled, y_pred_classes, average='macro')
recall = recall_score(y_test_type_resampled, y_pred_classes, average='macro')
f1 = f1_score(y_test_type_resampled, y_pred_classes, average='macro')

print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))


# Print the shapes of the resulting datasets after SMOTE
print("Training set shape for Machine failure prediction:", X_train_resampled.shape, y_train_failure_resampled.shape)
print("Testing set shape for Machine failure prediction:", X_test_resampled.shape, y_test_failure_resampled.shape)
print("Training set shape for Machine Failure Type prediction:", X_train_resampled.shape, y_train_type_resampled.shape)
print("Testing set shape for Machine Failure Type prediction:", X_test_resampled.shape, y_test_type_resampled.shape)


finalmodelMF = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_train_failure)), activation='softmax')
])

finalmodelMF.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = finalmodelMF.fit(X_train, y_train_failure, epochs=100, batch_size=32, validation_data=(X_test, y_test_failure))


# Evaluate the model on training data
train_loss, train_accuracy = finalmodelMF.evaluate(X_train, y_train_failure, verbose=0)
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))

# Evaluate the model on testing data
test_loss, test_accuracy = finalmodelMF.evaluate(X_test, y_test_failure, verbose=0)
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

# Predict on testing data
y_pred = finalmodelMF.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test_failure, y_pred_classes, average='macro')
recall = recall_score(y_test_failure, y_pred_classes, average='macro')
f1 = f1_score(y_test_failure, y_pred_classes, average='macro')

print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))


finalmodelMT = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_train_type)), activation='softmax')
])

finalmodelMT.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = finalmodelMT.fit(X_train, y_train_type, epochs=100, batch_size=32, validation_data=(X_test, y_test_type))


# Evaluate the model on training data
train_loss, train_accuracy = finalmodelMT.evaluate(X_train, y_train_type, verbose=0)
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))

# Evaluate the model on testing data
test_loss, test_accuracy = finalmodelMT.evaluate(X_test, y_test_type, verbose=0)
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

# Predict on testing data
y_pred = finalmodelMT.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test_type, y_pred_classes, average='macro')
recall = recall_score(y_test_type, y_pred_classes, average='macro')
f1 = f1_score(y_test_type, y_pred_classes, average='macro')

print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))

data = [[298.1, 308.6, 42.8, 0, -10.5]]
prediction = finalmodelMT.predict(data)

predicted_class = np.argmax(prediction)

print(predicted_class)

finalmodelMF.save('machine_failure_prediction_model.h5')

finalmodelMT.save('machine_failure_type_prediction_model.h5')

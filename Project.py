import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

datafile = 'C:/Users/dbsac/OneDrive/Documents/Centennial/Sem 3/COMP258/Project/Student data.csv'

# read column names
feature_columns = ['First Term Gpa', 'Second Term Gpa', 
                   'First Language', 'Funding', 'School', 
                   'FastTrack', 'Coop', 'Residency', 
                   'Gender', 'Previous Education', 'Age Group', 
                   'High School Average Mark', 'Math Score', 
                   'English Grade']

target_column = 'FirstYearPersistence'
all_columns = feature_columns + [target_column]

data = pd.read_csv(datafile, skiprows = 24, names=all_columns)

data.describe()
# total 1437 records
print('Number of missing record - ')
for column in data.columns:
    count = (data[column] == "?").sum()
    if count != 0:
        print(f"{column}: {count}")

'''
Number of missing record - 

First Term Gpa: 17
Second Term Gpa: 160
First Language: 111
Previous Education: 4
Age Group: 4
High School Average Mark: 743 
Math Score: 462
English Grade: 45'''

cat_columns = ['First Language', 'Funding', 'School', 
               'FastTrack', 'Coop', 'Residency', 
               'Gender', 'Previous Education', 'Age Group', 
               'English Grade']
num_columns = ['First Term Gpa', 'Second Term Gpa',
                'High School Average Mark', 'Math Score']
columns_to_impute = ['First Term Gpa', 'Second Term Gpa', 'First Language', 
                     'Previous Education', 'Age Group', 'High School Average Mark', 
                     'Math Score', 'English Grade']

# Replacing Missing Values
meanimputer = SimpleImputer(strategy='mean')
mostfrequentimputer = SimpleImputer(strategy='most_frequent')

data.replace('?', np.nan, inplace=True)

data[num_columns] = meanimputer.fit_transform(data[num_columns])
data[cat_columns] = mostfrequentimputer.fit_transform(data[cat_columns])

# Scale numeric data
scaler = StandardScaler()
data[num_columns] = scaler.fit_transform(data[num_columns])

'''
# One Hot Encoder (Should we do it?)
ohe_data = pd.get_dummies(data[cat_columns])
data = data.drop(cat_columns, axis=1)
data = pd.concat([data, ohe_data], axis=1)
'''
data = data.astype('float64')
X = data.drop(target_column, axis = 1)
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[14]))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(8, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)
model.save('project_model.h5')

# Evaluate the model using the validation dataset
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
epochs = range(1, len(training_loss) + 1)

# Plotting the loss on both training and validation
plt.figure(figsize=(8, 4))
plt.plot(epochs, training_loss, 'r', label='training loss')
plt.plot(epochs, validation_loss, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('# epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the accuracy on both training and validation
plt.figure(figsize=(8, 4))
plt.plot(epochs, training_accuracy, 'r', label='training accuracy')
plt.plot(epochs, validation_accuracy, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediction
y_pred = model.predict(X_test)

# Assuming a binary classification with a threshold of 0.5
y_pred_binary = (y_pred > 0.5).astype("int32")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy Score
acc_score = accuracy_score(y_test, y_pred_binary)
print("Accuracy Score:", acc_score)

# Additional metrics like Precision, Recall, F1-Score
class_report = classification_report(y_test, y_pred_binary)
print("Classification Report:")
print(class_report)

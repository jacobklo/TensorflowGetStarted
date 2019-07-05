import numpy as np
import pandas
import tensorflow as tf
import sklearn.model_selection



URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pandas.read_csv(URL)
print(dataframe.head())



train, test = sklearn.model_selection.train_test_split(dataframe, test_size=0.2)
train, val = sklearn.model_selection.train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')



def dataframe_to_dataset(dataframe, batch_size=32, shuffle=True ):
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 32
train_dataset = dataframe_to_dataset(train, batch_size)
val_dataset = dataframe_to_dataset(val, batch_size, shuffle=False)
test_dataset = dataframe_to_dataset(test, batch_size, shuffle=False)



for feature_batch, label_batch in train_dataset.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch )


example_batch = next(iter(train_dataset))[0]

def demo(feature_column):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())

age = tf.feature_column.numeric_column("age")
demo(age)

age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)



thal_hashed = tf.feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000)
demo(tf.feature_column.indicator_column(thal_hashed))

feature_columns = []

# numeric cols
for header in [ 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(tf.feature_column.numeric_column(header))

# bucketized cols
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = tf.feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = tf.feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = tf.feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)


# Create a feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# Create, compile, and train the model
model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=5
          )



loss, accuracy = model.evaluate(test_dataset)
print('Accuracy', accuracy)



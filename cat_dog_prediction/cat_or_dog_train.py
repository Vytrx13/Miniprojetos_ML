import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Part 1 - Data Preprocessing
train_datagen = ImageDataGenerator(rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])) # convolution
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) # pooling
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) # convolution
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) # pooling
cnn.add(tf.keras.layers.Flatten()) # flattening
cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) # full connection # 128 neurons 
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # output layer # this is binary classification

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# saving the model
cnn.save('cat_or_dog.h5')


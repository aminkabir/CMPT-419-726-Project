import os
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
from keras import optimizers

start = time.time()

#Define Path
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
test_path = 'dataset/test'
batch_size = 64
num_of_test_samples = 50000

#Define image parameters
img_width, img_height = 256, 256
batch_size = 16
samples_per_epoch = 1000
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 32
nb_filters3 = 16
nb_filters4 = 16
nb_filters5 = 16
conv1_size = 9
conv2_size = 7
conv3_size = 5
conv4_size = 5
conv5_size = 5
pool_size = 2
classes_num = 5

model = Sequential()

model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, kernel_initializer='he_normal', border_mode ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.2))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, kernel_initializer='he_normal', border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.2))

model.add(Convolution2D(nb_filters3, conv3_size, conv3_size, kernel_initializer='he_normal', border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.2))

model.add(Convolution2D(nb_filters4, conv4_size, conv4_size, kernel_initializer='he_normal', border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.2))

model.add(Convolution2D(nb_filters5, conv5_size, conv5_size, kernel_initializer='he_normal', border_mode ="same"))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(classes_num, activation='softmax'))


#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)
weights = model.get_weights()
model.set_weights(weights)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])


'''
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['L', 'N', 'P', 'R', 'V'])

model.evaluate_generator(generator=test_generator, verbose=1)

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_generator, num_of_test_samples // batch_size+1, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
print(len(test_generator.classes))
print(y_pred.shape)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['L', 'N', 'P', 'R', 'V']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
'''


#Prediction Function
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  #print(result)
  answer = np.argmax(result)
  if answer == 1:
    print("Predicted: Normal")
    return 'N'
  elif answer == 0:
    print("Predicted: LBBB")
    return 'L'
  elif answer == 2:
    print("Predicted: Paced")
    return 'P'
  elif answer == 3:
    print("Predicted: RBBB")
    return 'R'
  elif answer == 4:
    print("Predicted: PVC")
    return 'V'
  else:
    raise ValueError('Undefined answer')

y_preds = []
y_actual = []
#Walk the directory for every image
for i, ret in enumerate(os.walk(test_path)):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    print(ret[0] + '/' + filename)
    y_pred = predict(ret[0] + '/' + filename)
    y_preds.append(y_pred)
    y_actual.append(filename[0])
    print(" ")

print('Confusion Matrix')
print(confusion_matrix(y_actual, y_preds))
target_names = ['L', 'N', 'P', 'R', 'V']
print(classification_report(y_actual, y_preds, target_names=target_names))

#Calculate execution time
end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")

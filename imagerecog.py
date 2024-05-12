#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
from tensorflow.keras.utils import to_categorical


# In[ ]:


(X_train,y_train),(X_test, y_test)=cifar10.load_data()


# In[ ]:


X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255.0
X_test/=255.0


# In[ ]:


y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


# In[ ]:


model=Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.summary()


# In[ ]:


model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
    shuffle=True)


# In[ ]:


model_structure=model.to_json()
f=Path("model_structure.json")
f.write_text(model_structure)


# In[ ]:


model.save_weights("model_weight.h5")


# In[ ]:


from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np


# In[ ]:


class_labels=[
    "Planes",
    "car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]


# In[ ]:


f=Path("model_structure.json")
model_structure=f.read_text()


# In[ ]:


model=model_from_json(model_structure)


# In[ ]:


model.load_weights("model_weight.h5")


# In[ ]:


import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
img=load_img("dog.png",target_size=(32,32))
plt.imshow(img)


# In[ ]:


from tensorflow.keras.utils import img_to_array
image_to_test=img_to_array(img)


# In[ ]:


list_of_images=np.expand_dims(image_to_test,axis=0)


# In[ ]:


results=model.predict(list_of_images)


# In[ ]:


single_result=results[0]


# In[ ]:


most_likely_class_index=int(np.argmax(single_result))
class_likelihood=single_result[most_likely_class_index]


# In[ ]:


class_label=class_labels[most_likely_class_index]


# In[ ]:


print("This is a image is a {} likelihood: {:2f}".format(class_label, class_likelihood))


# In[ ]:





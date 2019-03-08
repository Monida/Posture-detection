    # -*- coding: utf-8 -*-
    """
    Created on Fri Mar  8 13:45:47 2019
    
    @author: Monica Daniela
    """
    #Import the libraries
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    
    #Create the CNN
    
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    
    #Fitt CNN to images
    from keras.preprocessing.image import ImageDataGenerator
    
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range=20,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    train_set = train_datagen.flow_from_directory('../photos_data/train_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')
    
    test_set = test_datagen.flow_from_directory('../photos_data/test_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

#Fit the classifier
classifier.fit_generator(train_set,
                         steps_per_epoch = 50,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 50)

#Evaluate the model


#Make single predictions
from keras.preprocessing import image
test_image = image.load_img('../photos_data/prediction/crouch (8).jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
    prediction = 'upright'
else:
    prediction = 'crouch'

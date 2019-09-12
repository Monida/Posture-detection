# Posture_detection
This repository contains the code of a simple binary classification model that takes pictures from upright and crouch postures and outputs the posture attained by the person in the photo. The end goal of this project is to create a more sofisticated posture clasifier that helps us assess the posture of patients with muskuloskeletal impairments. 

Some of the pictures used for training and testing the models where taken from the data collected for my [Master's thesis](https://tspace.library.utoronto.ca/handle/1807/70400), whose repository can be found [here](https://github.com/Monida/Masters-thesis), and some others from google images. 

The number of pictures used for each case is described below. Two equally distributed sets were used one for training and testing crouch posture and a second one for training and testing upright posture. 

- Training set: 19 pictures (master's data), 25 pictures (google images)
- Testing set: 5 pictures (master's data), 4 pictures (google images)

Since there were only a few pictures available, the ImageDataGenerator class from Keras was used to augment the pictures used for training the model.

The data is only availabe upon request and on a base-to-base case, since some of the pictures belong to the facility where the experiments took place.

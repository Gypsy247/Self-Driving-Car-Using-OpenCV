# Self-Driving-Car-using-OpenCV
The repository will contain multiple scripts that I learned and executed in order to understand the following concepts:
* NumPy
* OpenCV - to find and identify lane lines
* Using the concept of a Perceptron with Keras for Binary Classification
* Deep Neural Networks
* Self Driving Car Simulator using Behavioral Cloning, OpenCV and TensorFlow

# How to replicate the model and results:
* Install the Udacity Self Driving Car Simulator.
* Once installed, use the training mode on track 1 to drive through the tracks while trying to stay in the centre of the road. Speed is not important.
* For good results, I would recommend atleast 3-4 laps, followed by an equal number of laps in the opposite direction to avoid left turn bias.
* The car in the simulator is fitted with three cameras in the front, one for left, one for right and one for the centre. As you drive, the cameras capture and store screenshots of what the car sees. This is stored in a folder called "IMG".
* the "driving_log.csv" file is also created which records other data, out of which the steering angle is critical to our project. A steering angle with a negative value means a left turn and a positive value means a right turn.
* Once the training data is collected, you can push the images and the csv file to your github.
* Install the "requirements.txt" file
* Next, you can run the "behavioral_cloning.ipynb" notebook, which trains our model based on the images and the steering angles.
* In order to test the car and the model, make sure to save the drive.py file in the same location as the model file that gets downloaded after executing behavioral_cloning.
* Run the "drive.py" file, following which you restart the simulator and go into Autonomous mode, where the car starts driving on its own based on your trained model.

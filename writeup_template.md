# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image_for_writeup/architecture.jpg "Model Architecture"
[image2]: ./image_for_writeup/visualize_loss.png "Loss Visualization"
[image3]: ./image_for_writeup/original_image.png "Original Image"
[image4]: ./image_for_writeup/resize_image.png "Resized Image"
[image5]: ./image_for_writeup/crop_image.png "Cropping Image"
[image6]: ./image_for_writeup/flip_image.png "Flip Image"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* main.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
I also submitted a video which includes one full lap around the track in autonomous mode.
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model architecture is based on the paper [NVIDIA paper](https://arxiv.org/pdf/1604.07316v1.pdf).

![alt text][image1]

The elu activation function has been chosen for model non-linearity.
#### 2. Attempts to reduce overfitting in the model

I add a drop-out layer at the end to reduce overfitting.

The following figure shows the loss on the training and validation sets for each epoch.

![alt text][image2]

#### 3. Model parameter tuning

The model used an adam optimizer. When I use a pretrained model, I play the model with a low learning rate 0.0001.

The batchsize is 128 for saving the GPU memory.

Correction parameter is set to 0.2, which helps the vehicle to go back to the center of the track.


#### 4. Appropriate training data

To augment training data, all the three cameras and also flipping images are used.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I use the sample data for track 1 then I split the data set into training set and validation set (with parameter test_size=0.1).

For data augmentation I used all the cameras and also flipping images. 

One thing to remember is to convert the image from BGR to RGB format when using opencv to read in images.

To combat the overfitting, I added a drop out layer at the end.

I use mse loss function and adam optimizer to train the model. After 8 epochs the mse loss does not change much. So I save the model and run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (for example the second sharp corner). In order to improve the driving behavior in these cases, I generated some new images and used a pretrained model with a low learning rate to fine-tune the model.

At the end of the process, as showed in the video the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is based on the NVIDIA paper.
```sh
model = Sequential()
model.add(Lambda(lambda x:x/127.5 - 1.0, input_shape=(160,320,3)))
model.add(Lambda(resize))
model.add(Cropping2D(cropping=((20,10),(0,0))))

model.add(Convolution2D(24, 5, 5, border_mode='same', activation='elu'))
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(36, 5, 5, border_mode='same', activation='elu'))
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(48, 5, 5, border_mode='same', activation='elu'))
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))

model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(1))
```


#### 3. Creation of the Training Set & Training Process

As showed above, before training I also normalize the image and crop each image to focus on the portion that is useful for training. To speed up the training process, I also resize each image to a smaller size.


```sh
camera = random.choice(['center','left','right'])
if camera == 'center':
    img = cv2.imread(join(image_path, center.strip()))
    steering = steering
elif camera == 'left':
    img = cv2.imread(join(image_path, left.strip()))
    steering = steering + correction
elif camera == 'right':
    img = cv2.imread(join(image_path, right.strip()))
    steering = steering - correction

if random.choice([True, False]):
    img = cv2.flip(img, 1)
    steering *= -1.0
```

Batchsize is set to 128 and for each epoch I run 300 steps for training process.
After 8 epochs I save the model and run the simulator in the autonomous mode. The performance is good as showed in the video, the vehicle can drive by itself without leaving the track.

# Improvement
1. I cropped images before normalization to save normalization operations
```sh
model.add(Lambda(resize, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((20,10),(0,0))))
model.add(Lambda(lambda x:x/127.5 - 1.0))
```
2. Image Augmentation Visualization
Original Image:
![alt text][image3]
Resized Image:
![alt text][image4]
Cropped Image:
![alt text][image5]
Flipped Image:
![alt text][image6]
3. To improve the performance, I generated more images at the position where the vehicle is touching the lane. Now the vehicle can drive as expected. Pls check it in vedio1.mp4




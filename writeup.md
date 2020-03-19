# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./download.png
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

No preprocessing has been used except shuffling as the model got a accuracy of 96% with the asusual data set


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers looks like this.

```python
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W= tf.Variable(tf.truncated_normal(shape=(5,5,3,6), mean=mu, stddev=sigma))
    conv1_b= tf.Variable(tf.zeros(6))
    conv1= tf.nn.conv2d(x,conv1_W,strides=[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True) + conv1_b

    # Activation.
    conv1= tf.nn.relu(conv1)

    # Layer 2: Convolutional. Input = 28x28x6. Output = 14x14x10.
    conv3_W= tf.Variable(tf.truncated_normal(shape=(5,5,6,10), mean=mu, stddev=sigma))
    conv3_b= tf.Variable(tf.zeros(10))
    conv3= tf.nn.conv2d(conv1,conv3_W,strides=[1,2,2,1],padding='VALID',use_cudnn_on_gpu=True) + conv3_b

    # Activation.
    conv3= tf.nn.relu(conv3)

    # Layer 3: Convolutional. Input = 14x14x10. Output = 8x8x16.
    conv2_W= tf.Variable(tf.truncated_normal(shape=(5,5,10,16),mean=mu,stddev=sigma))
    conv2_b=tf.Variable(tf.zeros(16))
    conv2= tf.nn.conv2d(conv3,conv2_W,strides=[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True) + conv2_b
    
    # Activation.
    conv2= tf.nn.relu(conv2)

    # Pooling. Input = 8x8x16. Output = 4x4x16.
    conv2= tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    # Flatten. Input = 4x4x16. Output = 256.
    f= flatten(conv2)

    # Layer 4: Fully Connected. Input = 256. Output = 120.
    fc1_W= tf.Variable(tf.truncated_normal(shape=(int(np.shape(f)[1]),120),mean=mu,stddev=sigma))
    fc1_b= tf.Variable(tf.zeros(shape=120))
    fc1= tf.matmul(f,fc1_W) + fc1_b
    
    # Activation.
    fc1= tf.nn.relu(fc1)
    
    # Introduce Dropout after first fully connected layer
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 5: Fully Connected. Input = 120. Output = 100.
    fc2_W= tf.Variable(tf.truncated_normal(shape=(120,100),mean=mu,stddev=sigma))
    fc2_b= tf.Variable(tf.zeros(100))
    fc2= tf.matmul(fc1,fc2_W) + fc2_b
    
    # Activation.
    fc2= tf.nn.relu(fc2)
    
    # Layer 6: Fully Connected. Input = 100. Output = 84.
    fc4_W= tf.Variable(tf.truncated_normal(shape=(100,84),mean=mu,stddev=sigma))
    fc4_b= tf.Variable(tf.zeros(84))
    fc4= tf.matmul(fc2,fc4_W) + fc4_b
    
    # Activation.
    fc4= tf.nn.relu(fc4)
    
    # Layer 7: Fully Connected. Input = 84. Output = 43.
    fc3_W= tf.Variable(tf.truncated_normal(shape=(84,43),mean=mu,stddev=sigma))
    fc3_b= tf.Variable(tf.zeros(43))
    fc3= tf.matmul(fc4,fc3_W) + fc3_b
    logits=fc3
        
    return logits
```
 #### ->Details of the characteristics and qualities of the above architecture
  
  
  * The First three layers are convolutional layers. since the input to the first layer is a colored image, the depth of filter should also be 3 to match the rgb space. there are 6 filters of size 5*5*3 which are also the weights of first layer.stride is one and padding is valid
  
  * The second layer's input size is reduced to 28*28*6 because padding type is valid in previous layer and 5*5 filters removes two from each side, depth 6 is because of 6 filters used in last layer..stride is 2 and padding is valid, 10 filters of 5*5*10 are used in this layer 
  
  * The convolutional layer also follows the similar path, and has input size of 14x14x10 because of the previous layer's output given due stride of 2 and valid padding. 10 filters of 5*5*16 are used in this layer , producing the output of size 8*8*16
  
  * The max pooling with a stride of 2 produces the result of 4*4*16
  
  * this serves as the input to the fully connected layers after flattening,(4*4*16 = 256) with a drop out between 1st and 2nd fully connected layers.
  
  *there are four fully connected layers
  
  * the first layer takes 256 inputs and gives 120 
  
  * the second layer takes 120 inputs from last layer's output and produces 100
  
  * the third layer takes 100 inputs from last layer's output and produces 84
  
  * the final layer takes 84 inputs from last layer's output and produces 43 classes, logits which are then used with   softmax_cross_entropy_with_logits loss fucntion.
  
 
  
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with rate of 0.0009, a batch size of 128, 80 epochs. softmax_cross_entropy_with_logits was used as loss function.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.969 or 96.9%
* test set accuracy of 0.957 or 95.7%

The approach was experimental i started out with lenet and tried to train it. later I increased the model's layers one by one and tried them out. diffrerent learning rate's were used. soon it converged with the increase in layers.

* In the beginning i took the preprocessing steps like converting to gray scale but at a later stage found color would be a good feature for this project and it gave me better results.

* later at the model architecture defination stage, The first model i tried was LeNet architecure, which gave me nearly 89%.

* later on as i increased the number of fully connected layers too. by experimenting with many learning rate's like 0.001, 0.0001. 0.0009 and removing/adding layers, the above architecture, gave me the reuslts of near 96% validation accurary.

* I didn't do any experimentaion with optimizer as i always use adam.

*  number of epochs were also changed many times. 

* all the hyper parameters are as given above.

* finally to decrease possibility of over-fitting drop out was used.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

five German traffic signs that I found on the web are placed in the downloaded images directory.

### Discussion
* the images are already resized to 32*32 in height and width for the model.
* In most cases the size of these images would not be a problem to detect since datasets like these are easy one's and often used as a starter projects for deep learning.
* but in some cases like the above shown bicycle and pedestrian sign(see discussion in nottebook), the pixels get aggregated and this may pose as problems.
* I find using color images would be more effective

#### The submission documents the performance of the model when tested on the captured images (here 5 images).

* I wantedly took the images which are harder to detect since this will show how the size of the image probably leads to wrong activation.

* both the images which were wrongly detected in this case are discussed above in discussion. the probable reason is that pixels getting aggregated and which leads to wrong activation. if you look at the top 5 for this pedestrian image, it gives (11, 25, 27, 24, 23) as top indices , in which top1 is Right-of-way at the next intersection(index 11). if you were to compare and see these 2 images , the  Right-of-way at the next intersection sign (please google it for a quick look) looks like a man which confuses the classifier.

* similar is the case for bicycle which usually has two children in it running looks similar to bicycle sign when it's aggregated.

* The other two images even when reduced in size seems to be clear in shape.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| wildanimals      		| wildanimals  									| 
| pedestrains  			| Right-of-way at the next intersection			|
| aheadonly				| Ahead only									|
| noentry       		| No entry          			 				|
| bicycle		 	    | Children crossing     						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in Ipython notebook.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .96         			| wildanimals   								| 
| .011     				| pedestrain									|
| 1.00					| aheadonly 									|
| 1.00	      			| noentry		    			 				|
| .032				    | bicycle           							|





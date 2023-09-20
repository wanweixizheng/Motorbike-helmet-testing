675 final exam report
Caution:
The report contains only part of the code to illustrate the logic, please refer to the zip file for
detailed code
outline：
The whole project has four parts in total
Using yolo
Separate the motorcyclist's image from the video
and tag the motorcyclist image with or without a helmet
Using SVM
You will try to train the SVM with the above data using the appropriate kernel
Evaluate the performance of the SVM using a test set
Using Neural Networks
You will try to use a neural network, in this project I used CNN, trained with the data above
Evaluate the performance of the SVM using the test set
Summarize
Analysis svm and cnn result
Show what i learned form the project
(Part 1) Using yolo (object detector)：
steps:
1) Import the required library
2) Download the pre-trained YOLOv5 weights file. In this example, we will use YOLOv5s
model.
3) Read the video files
4) Apply object detectors (YOLOv5s) to each video in order to identify motorcyclists
4) For the identified motorcyclists, check if they are wearing a helmet(A motorcycle rider with
a helmet tag is 1, without a helmet is 0)
5) Save the results as images and corresponding labels
Readme:
object_detector.py and detect_motorcyclist.py in yolov5 flie
Yushen Jiao(20705457) and Xinguang Jiang(20701360) collaborated with me on this part
First please put the video you want to process into the folder (input_video)
Then put model = YOLOv5('your path/yolov5s.pt', device='cuda:0' if torch.cuda.is_available()
else 'cpu') in detect_motorcyclist.py Your path is the location of the yolov5s.pt file on your
computer
Then put video_folder = 'your path/input_video' in object_detector.py Your path is the
location of the input_video file on your computer
Same for output_folder = 'your path/output_data' Your path is the location of the
output_data file on your computer
if frame_count % (3*fps) == 0:. You can set how many seconds to extract a frame in this line,
depending on the computer's power, I use three seconds
Run object_detector.py, a set of images and the tags corresponding to this set of images will
be stored in output_data

(Part 2) Using SVM
Try to train and test with the data obtained from yolo using SVM with appropriate kernel.
Step:
1) Import the necessary libraries and define the preprocessing functions
2) Load the data and preprocess it
3) Normalize and dimensionality reduction of the data
4) Divide the data into training set, validation set and test set
5) Train the SVM model and perform tuning
6) Evaluate the model performance
Readme:
image_folder = 'Your path/output_data' Your path is the location of the output_data file on
your computer
labels_file = 'Your path/motorcycle_labels.npy' Your path is the location of the
motorcycle_labels.npy file on your computer
Run SVM.py

(Part 3) Using Neural Networks
step
1) Import the necessary libraries and define preprocessing functions
2) Create a dataset class and load the data in a format suitable for neural networks
3) Divide the data into training set, validation set and test set
4) Define the convolutional neural network architecture(I chose to use CNN)
5) Train the neural network model and perform tuning
6) Evaluate model performance
Readme
image_folder = 'Your path/output_data' Your path is the location of the output_data file on
your computer
labels_file = 'Your path/motorcycle_labels.npy' Your path is the location of the
motorcycle_labels.npy file on your computer
Run CNN.py


(part 4) Summarize
In this project, we used two approaches to solve the motorcycle helmet detection problem:
support vector machine (SVM) and convolutional neural network (CNN).
Data preprocessing and preparation:
Motorcycle drivers and helmets are detected using YOLOv5.
Extract images of motorcyclists and label them according to whether they wear helmets or
not.
Organize the images and labeled data into a suitable format.
Divide the data set into training, validation and test sets.
Support vector machine (SVM):
The test set was evaluated and an accuracy of 71% was obtained.
Convolutional Neural Network (CNN):
The test set was evaluated and 76% accuracy was obtained.
Some failure cases ：
For yolo：Some bikes are identified as motorbikes
1)open yolov5 run object_detector.py
2)Run SVM.py
3)Run CNN.py
Note:If you want to replace the test video please do so in the intput folder. Note that you
should empty the output folder each time you replace a file.
Comparison of results:
The accuracy of SVM is 71%, while the accuracy of CNN is 76%.
CNN performs better relative to SVM, which may be due to the ability of CNN to
automatically learn the feature representation of an image without the need to manually
design a feature extractor.
Analysis:
The results may not be ideal for optimal performance because the dataset may be small or
there may be issues with image quality and labeling.
The main reason may be that the yolo weights are poorly trained, so a lot of erroneous data
is obtained, and better yolo weights would have better results if they were trained.
In addition, the feature extractor and neural network architecture we use are relatively simple,
which may limit performance.
To improve the performance, we can consider using more data, more complex feature
extractors or deeper neural network architectures, or will use a combination of multiple
models.
What I learned：
It is Importance for data pre-processing, in this project, i resized and normalized the raw
images. The preprocessing step is critical to the performance of machine learning and deep
learning models. Proper preprocessing can help the model learn features better and improve
generalization.
I tried two different methods, SVM and CNN. SVM is a traditional machine learning method
that can handle linear indistinguishable problems by choosing the right kernel function. While
CNN, as a deep learning method, can automatically learn feature representations in images
and usually has better performance on image recognition tasks.
When training the model, the appropriate hyperparameter settings are crucial to the model
performance. For example, in SVM, we need to choose the appropriate kernel function and
tune the regularization parameters C and gamma values. In CNN, we need to determine the
network architecture, activation function, loss function, and optimizer, etc. In the project, I
divide the dataset into a training set, a validation set and a test set. By evaluating the model
performance on the validation set, I can monitor the overfitting or underfitting of the model.
Using a separate test set helps us to obtain the generalization performance of the model on
unknown data.
In conclued: attempts to apply machine learning and deep learning methods in real-world
projects can help us better understand the principles, strengths and weaknesses, and
applicability of these algorithms. Starting from real problems and gradually adjusting models
and parameters enables us to learn and grow in practice.

Cat & Dog Image Classifier

This project is a deep learning-based image classifier that distinguishes between images of cats and dogs. It includes a Convolutional Neural Network (CNN) model trained with Keras and TensorFlow, and a simple Flask web application that allows users to upload images for classification. The model achieved 89% accuracy on the test dataset.  
  
Table of Contents  
    &nbsp;&nbsp;&nbsp;&nbsp;• Project Overview  
    &nbsp;&nbsp;&nbsp;&nbsp;• Model Architecture    
    &nbsp;&nbsp;&nbsp;&nbsp;• Training Process    
    &nbsp;&nbsp;&nbsp;&nbsp;• Testing the Model  
    &nbsp;&nbsp;&nbsp;&nbsp;• Web Application    
    &nbsp;&nbsp;&nbsp;&nbsp;• Exception Handling & Logging  
    &nbsp;&nbsp;&nbsp;&nbsp;• Installation   
    &nbsp;&nbsp;&nbsp;&nbsp;• Usage  
    &nbsp;&nbsp;&nbsp;&nbsp;• Directory Structure  
    &nbsp;&nbsp;&nbsp;&nbsp;• Contributing  
    &nbsp;&nbsp;&nbsp;&nbsp;• License  

    
Project Overview  
  
The project involves building, training, and deploying a CNN model to classify images as either cats or dogs. The final solution includes:   
    &nbsp;&nbsp;&nbsp;&nbsp;1. A trained CNN model.  
    &nbsp;&nbsp;&nbsp;&nbsp;2. A testing script to evaluate model performance.  
    &nbsp;&nbsp;&nbsp;&nbsp;3. A Flask web application for user interaction.    
    &nbsp;&nbsp;&nbsp;&nbsp;4. Exception handling and logging mechanisms to ensure smooth operations.  

     
Model Architecture  
  
The model is a Sequential CNN consisting of the following layers:  
    &nbsp;&nbsp;&nbsp;&nbsp;• Input Layer: Accepts images resized to 128x128 with 3 color channels (RGB).   
    &nbsp;&nbsp;&nbsp;&nbsp;• Convolutional Layers: Four convolutional layers with increasing filter sizes (32, 64, 128, 256) and L2 regularization.  
    &nbsp;&nbsp;&nbsp;&nbsp;• Batch Normalization and ReLU Activation applied after each convolution.     
    &nbsp;&nbsp;&nbsp;&nbsp;• MaxPooling layers to reduce spatial dimensions.    
    &nbsp;&nbsp;&nbsp;&nbsp;• Dropout layers for regularization to prevent overfitting.    
    &nbsp;&nbsp;&nbsp;&nbsp;• Fully Connected Layer: Final Dense layer with softmax activation to classify images into two categories (Cat, Dog).  
    
Compilation Details:  
    &nbsp;&nbsp;&nbsp;&nbsp;• Optimizer: Adam    
    &nbsp;&nbsp;&nbsp;&nbsp;• Loss Function: Categorical Crossentropy    
    &nbsp;&nbsp;&nbsp;&nbsp;• Metrics: Accuracy  
    
Training Process  
    &nbsp;&nbsp;&nbsp;&nbsp;1. Data Preprocessing: Images are resized to 128x128 and normalized to values between [0, 1].      
    &nbsp;&nbsp;&nbsp;&nbsp;2. Data Augmentation: Techniques like rotation, flipping, zooming, brightness adjustment, and channel shifting are applied to enhance model robustness.    
    &nbsp;&nbsp;&nbsp;&nbsp;3. Dataset:    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◦ Can be obtained from https://www.microsoft.com/en-us/download/details.aspx?id=54765          
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◦ 80% of the images are used for training, rest is equally divided into two between test and validation sets.  
    &nbsp;&nbsp;&nbsp;&nbsp;4. Callbacks Used:    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◦ ModelCheckpoint: Saves model checkpoints every 2 epochs.        
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◦ EarlyStopping: Stops training if validation loss doesn't improve, restoring best weights.        
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◦ ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.  
        
Results  
    &nbsp;&nbsp;&nbsp;&nbsp;• Achieved 89% accuracy on the test dataset.    
    &nbsp;&nbsp;&nbsp;&nbsp;• Training and validation metrics are saved in history.txt.  

       
Testing the Model
  
A separate script is provided to evaluate the trained model on a test dataset:  
    &nbsp;&nbsp;&nbsp;&nbsp;1. Test Directory: path_to_test_dir     
    &nbsp;&nbsp;&nbsp;&nbsp;2. Usage: Run the script with the model path as an argument:    
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python test_model.py models/model.h5       
    &nbsp;&nbsp;&nbsp;&nbsp;3. The script will output:    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◦ Test Accuracy        
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◦ Test Loss        
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;◦ Predictions for each image  

  
Web Application  
  
A Flask web application allows users to upload images for classification. The app loads the trained model and predicts whether the uploaded image is a cat or a dog.

Features:  
    &nbsp;&nbsp;&nbsp;&nbsp;• Image Upload: Users can upload images via a simple UI.    
    &nbsp;&nbsp;&nbsp;&nbsp;• Prediction Output: Displays the predicted label (Cat or Dog).    
    &nbsp;&nbsp;&nbsp;&nbsp;• Unknown Detection: Images that don't match either class are labeled as 'UNKNOWN'.  

      
Exception Handling & Logging  
  
The project includes robust exception handling and logging mechanisms:  
    &nbsp;&nbsp;&nbsp;&nbsp;• Exception Handlers: Catch errors like file format issues, incorrect paths, or unsupported images.    
    &nbsp;&nbsp;&nbsp;&nbsp;• Logging: Tracks application events, model loading, and prediction processes to facilitate debugging.  
    
    
Installation  
    &nbsp;&nbsp;&nbsp;&nbsp;1. Clone the Repository:    
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;git clone https://gitlab.com/**repo-link      
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cd ***repo-link  
    &nbsp;&nbsp;&nbsp;&nbsp;2. Create a Virtual Environment (Optional but recommended):  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python3 -m venv venv   
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;source venv/bin/activate  # On Windows: venv\Scripts\activate      
    &nbsp;&nbsp;&nbsp;&nbsp;3. Install Required Packages:  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pip install -r requirements.txt  

         
Usage  
Train the Model  
If you want to retrain the model:  
    &nbsp;&nbsp;&nbsp;&nbsp;python train_model.py  

    
Test the Model  
python test_model.py models/model.h5  

    
Run the Web Application  
python app.py  

    
Directory Structure
<pre> <code> 
cat_dog_classifier/
├── corruption_log-file.txt
├── requirements.txt                    # Required Python packages
├── README.md                           # Project documentation
└── src/
    ├── train_model.py                  # Script to train the model
    ├── test_model.py                   # Script to test the model
    ├── show_output_of_each_layer.py    # Script to visualize intermediate layer 
    ├── draw_confusion_matrix.py
    ├── find_corrupted.py
    ├── distribute_images.py
    ├── combine_images.py
    └── history.txt  
├── models/                         	# Saved models
│   └── model.h5
└── website/ 
    ├── app.py                           # Flask web application
    ├── web.log 
    ├── LoggingConfig.py                       
    └── templates/                       # HTML templates for the Flask app
   	  ├──index.html
   	  └──result.html
    └── uploads/    
catndog_images/                          # Dataset directories
├── train/                         
    ├── CAT
    └── Dog
└── validation/ 
    ├── CAT
    └── Dog
└── test/ 
</code> </pre>
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.  
  
License  
This project is licensed under the MIT License.  
  
Happy Classifying! 🐱🐶

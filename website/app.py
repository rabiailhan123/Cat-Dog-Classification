from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from LoggingConfig import setup_logging     # LoggingConfig class is used to handle logging operations more centrally
import logging
import os

cwd = os.getcwd()                           # current working directory (for logging purposes)
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Test log: Logging setup complete.")
app = Flask(__name__)

# Load the trained model
try:
    model = load_model("../models_4/cat_dog_prediction_with_basic_cnn_epoch_11.h5")
    logger.info("Model loaded successfully from '../models_4/cat_dog_prediction_with_basic_cnn_epoch_11.h5'")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """
    The function `allowed_file` checks if a filename has an allowed extension based on a predefined list
    of allowed extensions.
    
    :param filename: The `filename` parameter is a string that represents the name of a file
    :return: The function `allowed_file` is returning a boolean value. It checks if the input `filename`
    contains a dot ('.') and if the file extension (the part after the last dot) is in the list of
    `ALLOWED_EXTENSIONS`.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict(img_path, model):
    """
    The function `model_predict` takes an image file path and a model, preprocesses the image, makes a
    prediction using the model, and returns a classification label based on the prediction threshold.
    
    :param img_path: The `img_path` parameter in the `model_predict` function is the file path to the
    image that you want to make a prediction on using the provided model. This function loads the image
    from the specified path, preprocesses it to match the input requirements of the model, makes a
    prediction using the
    :param model: The `model` parameter in the `model_predict` function refers to the machine learning
    model that you have trained and want to use for making predictions on images. This model should be
    loaded and passed as an argument to the function so that it can be used to predict the class of the
    image provided in
    :return: The function `model_predict` returns a string indicating whether the image is predicted to
    be a "Cat", "Dog", or "Unknown" based on the model's prediction probability.
    """
    try:
        logger.info(f"Preprocessing image: {img_path}")
        img = Image.open(img_path).resize(size = (128, 128))                # Adjust size to match the model input
        img = img.convert('RGB')                                            # convert the channel format to RGB
        img_array = image.img_to_array(img)                                 # convert the image into an array
        img_array = np.expand_dims(img_array, axis=0)   
        img_array /= 255.0                                                  # normalize the image
        
        prediction = model.predict(img_array)                               # make prediction on the image
        logger.info(f"Prediction result for {img_path}: {prediction[0][0]:.4f}")
        
        # Classification result based on the model output (can be adjusted)
        if prediction[0][0] > 0.505:
            return "Cat"
        elif prediction[0][0] <= 0.495:
            return "Dog"
        else:   # model is not sure
            return "Unknown"
    except Exception as e:
        logger.error(f"Error in predicting image {img_path}: {e}")
        return "Prediction Failed"

@app.route('/')
def index():
    """
    The `index()` function returns the rendered template "index.html".
    :return: The `index()` function is returning the result of calling the
    `render_template("index.html")` function, which renders an HTML template named
    "index.html".
    """
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    """
    The `upload` function checks if a file is uploaded, saves it if it is in an allowed format, and
    returns the prediction result using a model.
    :return: The `upload()` function returns either the result of the model prediction along with the
    filename in the rendered template 'result.html' if the file is successfully uploaded and is in an
    allowed format, or it returns a message stating "Invalid file format. Only PNG, JPG, and JPEG are
    allowed." if the file format is not allowed.
    """
    if 'file' not in request.files:
        logger.warning("No file part in the request.")
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("No file selected for uploading.")
        return "No file selected.", 400  # Bad Request
    
    if file and allowed_file(file.filename):
        try:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            logger.info(f"New file -{file.filename}- added to uploads.")

            result = model_predict(filepath, model)
            return render_template('result.html', result=result, filename=file.filename)
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            return "An error occurred while processing the file.", 500  # Internal Server Error
    else:
        invalid_ext = file.filename.rsplit('.', 1)[1] if '.' in file.filename else 'No extension'
        logger.warning(f"User uploaded an invalid file format: {invalid_ext}")
        return "Invalid file format. Only PNG, JPG, and JPEG are allowed.", 415  # Unsupported Media Type

@app.route('/uploads/<filename>')
def send_file(filename):
    """
    The function `send_file` is used to send a file from the 'uploads' directory with the specified
    filename.
    
    :param filename: The `filename` parameter in the `send_file` function is the name of the file that
    you want to send from the 'uploads' directory
    :return: The function `send_file` is returning the result of calling `send_from_directory('uploads',
    filename)`. This function is responsible for sending a file from a specific directory,
    'uploads', with the given filename.
    """
    try:
        logger.info(f"Sending file: {filename}")
        return send_from_directory('uploads', filename)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        abort(404)
    except Exception as e:
        logger.error(f"Error sending file {filename}: {e}")
        abort(500)

if __name__ == '__main__':
    # create uploads directory if does not exist
    if not os.path.exists('uploads'):
        logger.info(f"New directory -/uploads- created under {cwd}")
        try:
            os.makedirs('uploads')
        except Exception as e:
            logger.error(f"New directory -/uploads- creation failed.")
            raise

    logger.info("Starting Flask application...")
    app.run(debug=True, use_reloader=False)
    logger.info("Application has started successfully.")

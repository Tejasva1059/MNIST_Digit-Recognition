# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import tensorflow as tf
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "some_secret_key"

# Load the pre-trained MNIST model
model = tf.keras.models.load_model('models/mnist_model.h5')


@app.route('/')
def index():
    # Home page with options: "Use MNIST Dataset" and "Upload CSV File"
    return render_template('index.html')


@app.route('/mnist-dataset')
def mnist_dataset():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0  # Normalize

    # Sample test image
    sample_image = x_test[0].reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(sample_image))

    # Create the predicted digit snippet
    plt.imshow(sample_image[0], cmap='gray')
    plt.axis('off')  # Turn off the axis
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    digit_image_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Generate the accuracy and loss chart
    plt.style.use('dark_background')
    epochs = [1, 2, 3, 4, 5]  # Example epochs
    accuracy_values = [0.85, 0.9, 0.92, 0.95, 0.97]  # Example accuracy values
    loss_values = [0.15, 0.1, 0.08, 0.05, 0.03]  # Example loss values

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy_values, color='green', marker='o', label='Accuracy', linewidth=2)
    plt.plot(epochs, loss_values, color='red', marker='o', label='Loss', linewidth=2)

    plt.title('Model Accuracy vs Loss', fontsize=14, color='white')
    plt.xlabel('Epochs', fontsize=12, color='white')
    plt.ylabel('Values', fontsize=12, color='white')
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # Save the plot as image and convert it to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('result.html', prediction=prediction, plot_url=plot_url, digit_image_url=digit_image_url)


@app.route('/upload-csv', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            flash("No file selected.")
            return redirect(url_for('upload_csv'))

        try:
            # Read the uploaded CSV file
            df = pd.read_csv(file)
            # CSV should be 28x28 flattened pixel data
            images = df.values.reshape(-1, 28, 28, 1)
            images = images / 255.0  # Normalize images
            predictions = model.predict(images)
            predicted_labels = [np.argmax(pred) for pred in predictions]

            return render_template('result.html', predictions=predicted_labels)

        except Exception as e:
            flash(f"Error processing the file: {e}")
            return redirect(url_for('upload_csv'))

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)

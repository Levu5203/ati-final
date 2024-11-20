import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from keras.src.datasets import mnist
from keras.src.models.sequential import Sequential
from keras.src.layers.core.dense import Dense
from keras.src.layers.reshaping.flatten import Flatten
from keras.src.layers.core.input_layer import Input
from keras.src.utils.numerical_utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image


def train_model():
    global x_test, model, conf_matrix
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2)  # Use 1 epoch for simplicity
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Function to display a predicted image
def display_image():
    index = np.random.randint(0, len(x_test))
    test_image = x_test[index]
    prediction = model.predict(test_image.reshape(1, 28, 28))
    predicted_label = np.argmax(prediction)

    # Create a new window for the image
    img_window = tk.Toplevel(root)
    img_window.title("Predicted Image")

    # Display the image
    plt.figure(figsize=(4, 4))
    plt.imshow(test_image, cmap='gray')
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')

    # Embed Matplotlib figure in Tkinter
    canvas = FigureCanvasTkAgg(plt.gcf(), master=img_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Function to display the confusion matrix
def display_conf_matrix():
    # Create a new window for the confusion matrix
    conf_window = tk.Toplevel(root)
    conf_window.title("Confusion Matrix")

    # Display confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.arange(10))
    disp.plot(cmap='viridis', ax=ax)

    # Embed Matplotlib figure in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=conf_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Function to upload an image and predict
def upload_and_predict():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("PNG files", "*.png")])

    if file_path:
        try:
            # Open the image and resize it to 28x28
            img = Image.open(file_path).convert("L")  # Convert to grayscale
            img = img.resize((28, 28))
            img_array = np.array(img) / 255.0  # Normalize image

            # Reshape image to fit model input
            img_array = img_array.reshape(1, 28, 28)

            # Predict the image
            prediction = model.predict(img_array)
            predicted_label = np.argmax(prediction)

            # Create a new window for the uploaded image and prediction
            img_window = tk.Toplevel(root)
            img_window.title("Uploaded Image Prediction")

            # Display the uploaded image
            plt.figure(figsize=(4, 4))
            plt.imshow(img_array.reshape(28, 28), cmap='gray')
            plt.title(f"Predicted Label: {predicted_label}")
            plt.axis('off')

            # Embed Matplotlib figure in Tkinter
            canvas = FigureCanvasTkAgg(plt.gcf(), master=img_window)
            canvas.draw()
            canvas.get_tk_widget().pack()

        except Exception as e:
            messagebox.showerror("Error", f"Error loading or processing image: {str(e)}")

train_model()
#UI
root = tk.Tk()
root.title("Neural Network Model")
root.geometry("600x600")

# Create buttons
predict_button = tk.Button(root, text="Display Predicted Image", command=display_image, width=30, height=2)
predict_button.pack(pady=10)

matrix_button = tk.Button(root, text="Display Confusion Matrix", command=display_conf_matrix, width=30, height=2)
matrix_button.pack(pady=10)

upload_button = tk.Button(root, text="Upload Image and Predict", command=upload_and_predict, width=30, height=2)
upload_button.pack(pady=10)


# Run the Tkinter event loop
root.mainloop()
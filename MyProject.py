import cv2
import tensorflow as tf
import numpy as np

# Load the trained model in the recommended format
model = tf.keras.models.load_model('gender_classification.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define a function to preprocess the frames
def preprocess_frame(frame):
    # Resize the frame to match the input size of the model
    img_size = (64, 64)  # Change this according to your model's input size
    frame_resized = cv2.resize(frame, img_size)
    
    # Normalize the image (if needed, adjust according to your model's preprocessing)
    frame_normalized = frame_resized / 255.0
    
    # Add batch dimension
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    
    return frame_batch

# Define a function to display the predictions
def display_predictions(frame, prediction):
    label = "Male" if prediction[0][0] > 0.5 else "Female"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    text = f"{label}: {confidence:.2f}"
    
    # Put text on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Start the video loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame_preprocessed = preprocess_frame(frame)
    
    # Make predictions
    prediction = model.predict(frame_preprocessed)
    
    # Display predictions
    display_predictions(frame, prediction)
    
    # Show the frame
    cv2.imshow('Gender Classification', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

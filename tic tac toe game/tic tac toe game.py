import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as nppip
import matplotlib.pyplot as plt

# Load pre-trained ResNet model
base_model = tf.keras.applications.ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Load and preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract image features
def extract_image_features(img_path):
    img_array = preprocess_image(img_path)
    features = model.predict(img_array)
    return features

# Load your image
image_path = 'path/to/your/image.jpg'

# Example of extracting image features
image_features = extract_image_features(image_path)

# Define the captioning model
embedding_size = 300  # adjust as needed
vocab_size = 10000  # adjust based on your dataset
max_length = 20  # adjust based on your dataset

# Captioning model
input_image = Input(shape=(2048,))
image_embedding = Dense(embedding_size, activation='relu')(input_image)
input_caption = Input(shape=(max_length,))
caption_embedding = Embedding(vocab_size, embedding_size, input_length=max_length)(input_caption)
decoder = LSTM(256)(caption_embedding)
decoder = Dense(256, activation='relu')(decoder)
output = Dense(vocab_size, activation='softmax')(decoder)

captioning_model = Model(inputs=[input_image, input_caption], outputs=output)
captioning_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Example of using the captioning model
# You would need a dataset of image-caption pairs for training
# X_image, X_caption, y_caption = load_data()  # Implement this function
# captioning_model.fit([X_image, X_caption], y_caption, epochs=10, batch_size=32)

# Generate a caption for a new image
def generate_caption(image_features):
    start_token = 'startseq'
    caption = [start_token]
    
    for _ in range(max_length):
        sequence = [word_to_index[word] for word in caption]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        y_hat = captioning_model.predict([image_features, sequence], verbose=0)
        y_hat = np.argmax(y_hat)
        
        word = index_to_word[y_hat]
        caption.append(word)
        
        if word == 'endseq':
            break

    generated_caption = ' '.join(caption[1:-1])
    return generated_caption

# Example of generating a caption for a new image
generated_caption = generate_caption(image_features)
print("Generated Caption:", generated_caption)

# You'll need to adapt the code based on your dataset, training data, and specific requirements.

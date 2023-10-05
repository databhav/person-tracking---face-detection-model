import os 
import cv2
import face_recognition
import numpy as np 

# Load the pre-trained face recognition model
face_encodings = face_recognition.face_encodings

folder_path = "/home/anubhav/pyprojects/us/body/"

def compare_embeddings(embedding1, embedding2):
  # Calculate the cosine similarity between the two embeddings.
  cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
  # Return the cosine similarity.
  return cosine_similarity


# Create an empty list to store the embeddings.
embeddings = []

# Iterate over the images in the folder and extract the embeddings.
for filename in os.listdir(folder_path):
    # Load the image.
    image = face_recognition.load_image_file(os.path.join(folder_path, filename))
    # Detect the faces in the image.
    face_locations = face_recognition.face_locations(image)
    # If there is a face in the image, extract the embedding.
    if not face_locations:
        os.remove(os.path.join(folder_path, filename))
    else:
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        # Save the embedding to a file.
        with open(os.path.join(folder_path, filename + ".txt"), "wb") as f:
            f.write(face_encoding)

# Get a list of all the embedding files in the folder.
embedding_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".txt")]

# Create a dictionary to store the embeddings.
embeddings = {}

# Iterate over the embedding files and load the embeddings into the dictionary.
for embedding_file in embedding_files:
    with open(embedding_file, "rb") as f:
        embedding = np.fromfile(f, dtype=float)

        embeddings[embedding_file] = embedding

# Set the checkpoint embedding.
checkpoint_embedding = None

#embedding_files = sorted(embedding_files)

# Iterate over the embedding files.
for embedding_file in embedding_files:

    # If there is no checkpoint embedding, set it to the current embedding.
    if checkpoint_embedding is None:
        checkpoint_embedding = embeddings[embedding_file]

    # If the embedding file doesn't exist, continue to the next embedding file.
    if not os.path.exists(embedding_file):
        continue
        
    # Compare the current embedding to the checkpoint embedding.
    cosine_similarity = compare_embeddings(embeddings[embedding_file], checkpoint_embedding)

    # If the current embedding is similar to the checkpoint embedding, delete the current embedding file and the corresponding image file.
    if cosine_similarity > 0.90:
        os.remove(embedding_file)
        os.remove(os.path.join(folder_path, embedding_file.replace(".txt", "")))

    # Otherwise, the current embedding becomes the new checkpoint embedding.
    else:
        checkpoint_embedding = embeddings[embedding_file]


def extract_color_histogram(image):
  """Extracts a color histogram from an image.

  Args:
    image: A numpy array representing the image.

  Returns:
    A numpy array representing the color histogram.
  """

  # Quantize the image.
  color_bins = 256
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  image = np.round(image * color_bins / 180).astype(np.uint8)

  # Calculate the number of pixels in the image with each color value.
  histogram = np.zeros((color_bins,), dtype=int)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      histogram[image[i, j, 0]] += 1

  # Normalize the histogram.
  histogram = histogram / np.sum(histogram)

  return histogram

def compare_histograms(histogram1, histogram2):
  cosine_similarity = np.dot(histogram1, histogram2) / (np.linalg.norm(histogram1) * np.linalg.norm(histogram2))

  return cosine_similarity

# Iterate over the images in the folder and extract the embeddings.
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        hist = extract_color_histogram(image)
        with open(os.path.join(folder_path, filename + "-h.txt"), "wb") as f:
            f.write(hist)

# Get a list of all the embedding files in the folder.
hist_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith("-h.txt")]

# Create a dictionary to store the embeddings.
hists = {}

# Iterate over the embedding files and load the embeddings into the dictionary.
for hist_file in hist_files:
    with open(hist_file, "rb") as f:
        hist1 = np.fromfile(f, dtype=float)
        hists[hist_file] = hist1

# Set the checkpoint embedding.
checkpoint_hist = None

hist_files = sorted(hist_files)
hist_files

# Iterate over the embedding files.
for hist_file in hist_files:

    # If there is no checkpoint embedding, set it to the current embedding.
    if checkpoint_hist is None:
        checkpoint_hist = hists[hist_file]

    # If the embedding file doesn't exist, continue to the next embedding file.
    if not os.path.exists(hist_file):
        continue
        
    # Compare the current embedding to the checkpoint embedding.
    cosine_similarity = compare_histograms(hists[hist_file], checkpoint_hist)

    # If the current embedding is similar to the checkpoint embedding, delete the current embedding file and the corresponding image file.
    if cosine_similarity > 0.79:
        os.remove(hist_file)
        os.remove(os.path.join(folder_path, hist_file.replace("-h.txt", "")))
        os.remove(os.path.join(folder_path, hist_file.replace("-h", "")))
        
        

    # Otherwise, the current embedding becomes the new checkpoint embedding.
    else:
        checkpoint_hist = hists[hist_file]
import streamlit as st
import os
import csv
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Set the path to your service account key JSON file in Downloads
service_account_key_path = "/Users/bhavinvullli/Downloads/textembed-1c88c265e2aa.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path


# Function to get access token
def get_access_token():
    # Load credentials from the environment variable
    credentials = service_account.Credentials.from_service_account_file(
        service_account_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    # Refresh the credentials (if necessary)
    credentials.refresh(Request())
    return credentials.token


# Function to fetch embeddings from Vertex AI
def get_embeddings(text):
    access_token = get_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    # Replace with your specific Vertex AI endpoint
    endpoint = "https://us-central1-aiplatform.googleapis.com/v1/projects/textembed/locations/us-central1/publishers/google/models/textembedding-gecko@003:predict"
    data = {
        "instances": [
            {"content": text}
        ]
    }
    # Make POST request to Vertex AI prediction endpoint
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        embeddings = response.json()["predictions"][0]["embeddings"]["values"]
        return embeddings
    else:
        st.error(f"Failed to fetch embeddings: {response.text}")


# Function to calculate cosine similarity
def cosine_similarity_score(embeddings1, embeddings2):
    # Reshape embeddings arrays to ensure they are 2D arrays
    embeddings1 = np.array(embeddings1).reshape(1, -1)
    embeddings2 = np.array(embeddings2).reshape(1, -1)
    # Calculate cosine similarity
    similarity_score = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity_score


# Function to save data to CSV
def save_to_csv(word1, word2, embeddings1, embeddings2, similarity_score):
    csv_filename = "words.csv"
    # Check if CSV file exists, create header if it doesn't exist
    csv_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not csv_exists:
            writer.writerow(['word1', 'word2', 'embedding1', 'embedding2', 'similarity'])
        writer.writerow([word1, word2, embeddings1, embeddings2, similarity_score])


# Main function to run Streamlit app
def main():
    st.title("Text Embeddings App")

    # Input boxes for user input
    word1 = st.text_input("Enter word 1", "")
    word2 = st.text_input("Enter word 2", "")

    # Button to trigger embeddings calculation
    if st.button("Calculate Embeddings and Similarity"):
        embeddings1 = get_embeddings(word1)
        embeddings2 = get_embeddings(word2)

        if embeddings1 and embeddings2:
            similarity_score = cosine_similarity_score(embeddings1, embeddings2)
            st.write(f"Similarity Score between '{word1}' and '{word2}': {similarity_score:.4f}")

            # Save data to CSV
            save_to_csv(word1, word2, embeddings1, embeddings2, similarity_score)
            st.success("Data saved to CSV successfully!")


if __name__ == "__main__":
    main()

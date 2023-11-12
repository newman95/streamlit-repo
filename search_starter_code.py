import streamlit as st
import numpy as np
import numpy.linalg as la
import pickle

"""
This is a starter code for Assignment 0 of the course, "Hands-on Master Class on LLMs and ChatGPT | Autumn 2023"
taught by Dr. Karthik Mohan.

Computes closest category of a given word or sentence input into a search bar.
The search is implemented through streamlit and can be hosted as a "web app" on the cloud through streamlit as well
Example webpage and search demo: searchdemo.streamlit.app
"""


# Compute Cosine Similarity
def cosine_similarity(x, y):
    x_arr = np.array(x)
    y_arr = np.array(y)

    ############################
    ### WRITE YOUR CODE HERE ###
    ############################
    # return 

    dot_product = np.dot(x_arr, y_arr)
    matrix_x = np.linalg.norm(x_arr)
    matrix_y = np.linalg.norm(y_arr)
    similarity = dot_product / (matrix_x * matrix_y)
    return similarity


# Function to Load Glove Embeddings
def load_glove_embeddings(glove_path="Data/embeddings.pkl"):
    """
    First step: Download the 50d Glove embeddings from here - https://www.kaggle.com/datasets/adityajn105/glove6b50d
    Second step: Format the glove embeddings into a dictionary that goes from a word to the 50d embedding.
    Third step: Store the 50d Glove embeddings in a pickle file of a dictionary.
    Now load that pickle file back in this function
    """
    embeddings = format_glove_embeddings("archive/glove.6B.50d.txt")
    with open(glove_path, "wb") as f:
        pickle.dump(embeddings, f)

    with open(glove_path, "rb") as f:
        embeddings_dict = pickle.load(f)

    return embeddings_dict


def format_glove_embeddings(glove_file_path):
    embeddings_dict = {}

    with open(glove_file_path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            embedding = [float(val) for val in values[1:]]
            embeddings_dict[word] = embedding

    return embeddings_dict


# Get Averaged Glove Embedding of a sentence
def averaged_glove_embeddings(sentence, embeddings_dict):
    """
    Simple sentence embedding: Embedding of a sentence is the average of the word embeddings
    """
    words = sentence.split(" ")
    glove_embedding = np.zeros(50)
    count_words = 0

    ############################
    ### WRITE YOUR CODE HERE ###
    ############################


    for word in words:
        if word in embeddings_dict:
            glove_embedding += embeddings_dict[word]
            count_words += 1
    if count_words > 0:
        averaged_embedding = glove_embedding / count_words
    else:
        averaged_embedding = np.zeros(50)
    return averaged_embedding


# Load glove embeddings
glove_embeddings = load_glove_embeddings()

# Gold standard words to search from
gold_words = ["flower", "mountain", "tree", "car", "building"]
word_url=["https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTX8SDxrprQPavlXox8jmsd6ajpMpwPd7xURMPOg2e9Pawngz_NG7ptyCyFz5eogFyKD3M&usqp=CAU",
          "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSZ5HBYnAQ41oS77Fovv18_P2bPENluPd1xd_ubPd_kYyTcVuu-DpeVOYouHP6oZfcuQzM&usqp=CAU",
          "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT2wqbvAodhHRlRcwjS82ul8OoxacKtvleMQyqVkOPuFDo4esIrrDNzQCbN6_kSNDArUUc&usqp=CAU",
          "https://carwow-uk-wp-3.imgix.net/18015-MC20BluInfinito-scaled-e1666008987698.jpg",
          "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSK4M5a4wdE3yU4FG1YYMrqwy8cyWR1Yj7KRg06mxNW9Cn6o5r5WoTKVZsgBi1AOzxZSVk&usqp=CAU"]
# Text Search
st.title("Search Based Retrieval Demo")
st.subheader("Pass in an input word or even a sentence (e.g. jasmine or mount adams)")
text_search = st.text_input("", value="")

# Find closest word to an input word
if text_search:
    text_search = text_search.lower()
    input_embedding = averaged_glove_embeddings(text_search, glove_embeddings)
    cosine_sim = {}
    for index in range(len(gold_words)):
        cosine_sim[index] = cosine_similarity(input_embedding, glove_embeddings[gold_words[index]])

    ############################
    ### WRITE YOUR CODE HERE ###
    ############################

    # Sort the cosine similarities
    # sorted_cosine_sim =
    sorted_cosine_sim = sorted(cosine_sim.items(), key=lambda item: item[1], reverse=True)

    st.write("(My search uses glove embeddings)")
    st.write("Closest word I have between flower, mountain, tree, car and building for your input is: ")
    st.image(word_url[sorted_cosine_sim[0][0]])
    st.write("")

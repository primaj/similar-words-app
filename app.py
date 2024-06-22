import streamlit as st
import gensim.downloader as api
from gensim.models import KeyedVectors


@st.cache_resource
def load_model(model_name='word2vec-google-news-300'):
    """
    Load a pre-trained word vector model from gensim data repository and cache it.

    :param model_name: Name of the pre-trained model to load.
    :return: Loaded word vector model.
    """
    model = api.load(model_name)
    return model


def find_similar_words(model, word, topn=10):
    """
    Find the top N most similar words to the given word using the pre-trained model.

    :param model: Pre-trained word vector model.
    :param word: Word for which to find similar words.
    :param topn: Number of top similar words to return.
    :return: List of tuples containing similar words and their similarity scores.
    """
    try:
        similar_words = model.most_similar(word, topn=topn)
    except KeyError:
        st.error(f"The word '{word}' is not in the vocabulary.")
        similar_words = []
    return similar_words


def filter_similar_words(similar_words, input_word):
    """
    Filter out words with underscores and words that are the same as the input word (case insensitive).

    :param similar_words: List of tuples containing similar words and their similarity scores.
    :param input_word: The input word to compare against.
    :return: Filtered list of similar words.
    """
    input_word_lower = input_word.lower()
    filtered_words = [(word, score) for word, score in similar_words if '_' not in word and word.lower() != input_word_lower]
    return filtered_words


# Streamlit app
st.title("Word Similarity Finder")

model_name = st.selectbox('Select a pre-trained model', ['word2vec-google-news-300', 'glove-wiki-gigaword-50'])
word = st.text_input("Enter a word:")
topn = st.number_input("Enter the number of similar words to find:", min_value=1, value=10)

if st.button("Find Similar Words"):
    if word:
        with st.spinner('Loading model and finding similar words...'):
            model = load_model(model_name)
            similar_words = find_similar_words(model, word, topn)
            filtered_words = filter_similar_words(similar_words, word)

            if filtered_words:
                st.success(f"Top {topn} filtered words similar to '{word}':")
                for similar_word, score in filtered_words:
                    st.write(f"{similar_word}: {score:.4f}")
            else:
                st.warning(f"No filtered similar words found for '{word}'.")
    else:
        st.warning("Please enter a word to search.")

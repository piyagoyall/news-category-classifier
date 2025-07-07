<<<<<<< HEAD
import streamlit as st
import pickle
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the model and tools
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit configuration
st.set_page_config(page_title="News Classifier", layout="centered")

# Sidebar for project info
with st.sidebar:
    st.title("📘 About Project")
    st.markdown("""
    **Made by:** Piya Goyal  
    **Model Used:** Logistic Regression  
    **Goal:** Classify BBC news into correct category
    
    **Categories**:
    - 📊 Business
    - 🎬 Entertainment
    - 🏛️ Politics
    - 🏅 Sport
    - 💻 Tech
    """)

# Main UI
st.markdown("<h1 style='text-align: center;'>📰 News Category Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Paste a news headline or content below to predict its category.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input
user_input = st.text_area("📝 Enter News Headline or Content", height=150, help="Paste a news headline or article snippet.")

# Custom CSS for the button color and output box
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007bff;
        color: #ffffff;
        border: none;
        padding: 0.5em 2em;
        font-size: 16px;
        border-radius: 8px;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
    }
    .result-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 6px solid #2196f3;
        padding: 15px;
        border-radius: 10px;
        color: #0d47a1;
        font-size: 18px;
        font-weight: 500;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Predict Button
if st.button("🚀 Predict Category"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news headline or content.")
    else:
        with st.spinner("Analyzing headline..."):
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            category = label_encoder.inverse_transform([prediction])[0]

        # Styled result
        st.markdown(f"""
        <div class='result-box'>
        🎯 Predicted Category: <strong>{category.upper()}</strong>
        </div>
        """, unsafe_allow_html=True)

        category_info = {
            "business": "📊 Related to economy, finance, companies, etc.",
            "entertainment": "🎬 Movies, TV, celebrities, shows, etc.",
            "politics": "🏛️ Government, laws, elections, etc.",
            "sport": "🏅 Sports events, athletes, scores, etc.",
            "tech": "💻 Technology, gadgets, software, etc."
        }

        st.markdown(f"**About this category:** {category_info.get(category, 'No details available')}.")

# Footer
st.markdown("---")
=======
import streamlit as st
import pickle
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the model and tools
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit configuration
st.set_page_config(page_title="News Classifier", layout="centered")

# Sidebar for project info
with st.sidebar:
    st.title("📘 About Project")
    st.markdown("""
    **Made by:** Piya Goyal  
    **Model Used:** Logistic Regression  
    **Goal:** Classify BBC news into correct category
    
    **Categories**:
    - 📊 Business
    - 🎬 Entertainment
    - 🏛️ Politics
    - 🏅 Sport
    - 💻 Tech
    """)

# Main UI
st.markdown("<h1 style='text-align: center;'>📰 News Category Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Paste a news headline or content below to predict its category.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input
user_input = st.text_area("📝 Enter News Headline or Content", height=150, help="Paste a news headline or article snippet.")

# Custom CSS for the button color and output box
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007bff;
        color: #ffffff;
        border: none;
        padding: 0.5em 2em;
        font-size: 16px;
        border-radius: 8px;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
    }
    .result-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 6px solid #2196f3;
        padding: 15px;
        border-radius: 10px;
        color: #0d47a1;
        font-size: 18px;
        font-weight: 500;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Predict Button
if st.button("🚀 Predict Category"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news headline or content.")
    else:
        with st.spinner("Analyzing headline..."):
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            category = label_encoder.inverse_transform([prediction])[0]

        # Styled result
        st.markdown(f"""
        <div class='result-box'>
        🎯 Predicted Category: <strong>{category.upper()}</strong>
        </div>
        """, unsafe_allow_html=True)

        category_info = {
            "business": "📊 Related to economy, finance, companies, etc.",
            "entertainment": "🎬 Movies, TV, celebrities, shows, etc.",
            "politics": "🏛️ Government, laws, elections, etc.",
            "sport": "🏅 Sports events, athletes, scores, etc.",
            "tech": "💻 Technology, gadgets, software, etc."
        }

        st.markdown(f"**About this category:** {category_info.get(category, 'No details available')}.")

# Footer
st.markdown("---")
>>>>>>> a7e3ac321c25598f579895277d81b01dfbb95e97
st.markdown("<p style='text-align: center; font-size: 14px;'>📚 Project by Piya Goyal | B.Tech CSE</p>", unsafe_allow_html=True)
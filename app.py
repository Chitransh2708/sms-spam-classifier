import streamlit as st
import pickle
import string

# ---------------------------
# Simple Stemmer (lightweight)
# ---------------------------
class SimpleStemmer:
    def stem(self, word):
        suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
        for suf in suffixes:
            if word.endswith(suf) and len(word) > len(suf)+2:
                return word[:-len(suf)]
        return word

ps = SimpleStemmer()

# ---------------------------
# Stopwords (common English)
# ---------------------------
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','any','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
}

# ---------------------------
# Text Preprocessing Function
# ---------------------------
def transform_text(text):
    text = text.lower()
    words = text.split()
    cleaned = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation) and w not in STOPWORDS]
    stemmed = [ps.stem(w) for w in cleaned]
    return " ".join(stemmed)

# ---------------------------
# Load vectorizer and model
# ---------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message to classify:")

if st.button('Predict'):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)
    
    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])
    
    # 3. Predict
    result = model.predict(vector_input)[0]
    
    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

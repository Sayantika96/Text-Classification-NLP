def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W',' ', text)
    text = re.sub(r'\s+',' ',text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)
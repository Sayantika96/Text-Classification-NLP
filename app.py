from flask import Flask, request, jsonify
import pandas as pd
from text_preprocessing import preprocess_text  # Import your preprocessing function here

app = Flask(__name__)

@app.route('/api/preprocess', methods=['POST'])
def preprocess_text():
    try:
        data = request.get_json()
        text = data['text']
        
        # Call the text preprocessing function on the input text
        processed_text = preprocess_text(text)
        
        return jsonify({'processed_text': processed_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

from flask import Flask, render_template, request, jsonify, send_from_directory
from RAG import RAGSystem  # Import your existing RAG code
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Serve static files directly
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('../static', filename)

# Initialize RAG system
rag = RAGSystem()
rag._initialize()

# Use only one home route
@app.route('/')
def home():
    return render_template('index.html')  # Assuming you want to serve from templates

@app.route('/wakeup', methods=['POST'])
def wake_up():
    if request.headers.get('X-API-KEY') != os.getenv('WEBHOOK_KEY'):
        return jsonify({"error": "Unauthorized"}), 403

    question = request.json.get('question')
    answer = rag.query(question)  

    return jsonify({"response": answer})

@app.before_request
def check_auth():
    if request.endpoint != 'wakeup':
        api_key = request.headers.get('X-API-KEY')
        if api_key != os.getenv('API_KEY'):
            return jsonify({"error": "Unauthorized"}), 403

@app.route('/process_input', methods=['POST'])
def process_input():
    data = request.get_json()
    question = data['question']

    try:
        answer = rag.query(question)
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

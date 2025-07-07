from flask import Flask, render_template, request, jsonify, send_from_directory
from RAG import RAGSystem  # Import your existing RAG code
import os

app = Flask(__name__, static_folder='../static')

# Serve static files directly
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('../static', filename)

# Initialize RAG system
rag = RAGSystem()

# Your existing routes
@app.route('/')
def home():
    return send_from_directory('../static', 'templates/index.html')


@app.route('/wakeup', methods=['POST'])
def wake_up():
    # Check API key
    if request.headers.get('X-API-KEY') != os.getenv('WEBHOOK_KEY'):
        return jsonify({"error": "Unauthorized"}), 403
    
    # Process request
    question = request.json.get('question')
    answer = rag_app.query(question)
    
    return jsonify({"response": answer})

@app.before_request
def check_auth():
    if request.endpoint != 'wakeup':
        api_key = request.headers.get('X-API-KEY')
        if api_key != os.getenv('API_KEY'):
            return jsonify({"error": "Unauthorized"}), 403

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    data = request.get_json()
    question = data['question']  # Changed from 'prompt' to 'question'
    
    try:
        answer = rag.query(question)  # Using the correct method name
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
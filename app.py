from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
with open('rag_search_model.pkl', 'rb') as f:
    qa_chain = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    result = qa_chain.run(query)
    return jsonify({'answer': result})

if __name__ == '__main__':
    app.run(debug=True)
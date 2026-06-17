import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# On ouvre les CORS à 100% pour bloquer les erreurs de navigateur de echosai.ca
CORS(app, resources={r"/*": {"origins": "*"}})

# Utilisation stricte et unique de la clé FREE
API_KEY_FREE = os.getenv("API_KEY_FREE", "").strip()
client = genai.Client(api_key=API_KEY_FREE)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        # Configuration minimaliste sans aucun outil externe
        config = types.GenerateContentConfig(
            max_output_tokens=1024
        )

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=user_message,
            config=config
        )

        echo_text = response.text if (response and hasattr(response, 'text')) else "Erreur de génération"
        return jsonify({"response": echo_text})

    except Exception as e:
        return jsonify({"response": f"Erreur Flask Free: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
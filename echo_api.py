import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# On récupère UNIQUEMENT la clé Free
API_KEY_FREE = os.getenv("API_KEY_FREE", "").strip()
print(f"📡 [INIT] Clé API Free détectée : {API_KEY_FREE[:6]}...{API_KEY_FREE[-4:] if len(API_KEY_FREE) > 4 else ''}")

# Initialisation unique du client Free
client_gemini_free = genai.Client(api_key=API_KEY_FREE) if API_KEY_FREE else None

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "debug_mode", "engine": "gemini-free-only"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    if not client_gemini_free:
        print("❌ [ERREUR CRITIQUE] Le client API Free n'est pas initialisé. Vérifie ton .env")
        return jsonify({"action": None, "response": "Erreur : Clé API Free manquante ou mal configurée."}), 500

    try:
        data = request.json or {}
        user_message = data.get("message", "")
        
        print(f"📥 [REQUÊTE] Message utilisateur reçu : '{user_message}'")

        # Configuration ultra-simple sans formatage JSON forcé ni Google Search
        # On veut juste voir si le modèle de base accepte de répondre avec cette clé
        response = client_gemini_free.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_message,
            config=types.GenerateContentConfig(
                max_output_tokens=1024
            )
        )

        # On affiche la réponse brute de Google directement dans les logs de Railway
        print(f"📦 [REPONSE BRUTE GOOGLE] : {response}")

        if response and hasattr(response, 'text') and response.text:
            print(f"✅ [SUCCÈS] Texte extrait : {response.text}")
            return jsonify({"action": None, "response": response.text})
        else:
            print("⚠️ [ATTENTION] Réponse reçue de Google, mais le champ .text est vide ou introuvable.")
            return jsonify({"action": None, "response": "Le sillage d'Echo est vide. Réessaie."})

    except Exception as e:
        # ICI : On capture l'erreur exacte renvoyée par Google (Ex: Bad Request, Invalid API Key, etc.)
        print(f"💥 [CRASH DETECTÉ] Erreur brute de l'appel Google : {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "action": None, 
            "response": f"Détection du blocage Free brute : {str(e)}"
        }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY_FREE = os.getenv("API_KEY_FREE", "").strip()
API_KEY_PAID = os.getenv("API_KEY_PAID", "").strip()

print(f"📡 [INIT] Clé Free : {API_KEY_FREE[:6]}... | Clé Paid : {API_KEY_PAID[:6]}...")

client_gemini_free = genai.Client(api_key=API_KEY_FREE) if API_KEY_FREE else None
client_gemini_paid = genai.Client(api_key=API_KEY_PAID) if API_KEY_PAID else None

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "production", "engine": "hybrid-cascade-fixed"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        user_message = data.get("message", "")
        print(f"📥 [CHAT] Message reçu : '{user_message[:50]}'")

        # Configuration standard sans forcer le type MIME JSON (Rappel : Google Search + JSON strict = crash)
        config_standard = types.GenerateContentConfig(
            max_output_tokens=1024,
            tools=[{"google_search": {}}]
        )

        # 1. Tentative sur le client gratuit avec le BON ID trouvé !
        if client_gemini_free:
            try:
                print("🔄 [TRY] Appel Clé Free -> gemini-3.1-flash-lite")
                response = client_gemini_free.models.generate_content(
                    model="gemini-3.1-flash-lite", # ── L'ID exact que tu as débusqué ! 🚀
                    contents=user_message,
                    config=config_standard
                )
                if response and hasattr(response, 'text') and response.text:
                    print("✅ [SUCCÈS FREE] Réponse générée par la clé gratuite !")
                    return jsonify({"action": None, "response": response.text})
            except Exception as free_err:
                print(f"⚠️ [FREE FAIL] Échec clé gratuite (Probable 429 ou Quota) : {free_err}")

        # 2. Filet de secours automatique sur la clé payante si le gratuit sature
        if client_gemini_paid:
            try:
                print("🚨 [FILET ACTIVÉ] Bascule sur la Clé Payante de secours...")
                response = client_gemini_paid.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=user_message,
                    config=config_standard
                )
                if response and hasattr(response, 'text') and response.text:
                    print("✅ [SUCCÈS FILET] Réponse générée par la clé payante !")
                    return jsonify({"action": None, "response": response.text})
            except Exception as paid_err:
                print(f"❌ [CRITIQUE] Échec clé payante : {paid_err}")

        return jsonify({"action": None, "response": "Ouf, mon sillage sature sous l'affluence. Laisse-moi une petite seconde ! 😎"})

    except Exception as e:
        print(f"💥 Erreur générale backend : {e}")
        return jsonify({"action": None, "response": "Système instable, réessaie !"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
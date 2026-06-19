import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── RÉCUPÉRATION DES VARIABLES ────────────────────────────────────────────────
API_KEY_FREE          = os.getenv("API_KEY_FREE", "").strip()
API_KEY_PAID          = os.getenv("API_KEY_PAID", "").strip()
OPENROUTER_API_KEY    = os.getenv("OPENROUTER_API_KEY", "").strip()
GITHUB_API_KEY        = os.getenv("GITHUB_API_KEY", "").strip()
NVIDIA_API_KEY        = os.getenv("NVIDIA_API_KEY", "").strip()
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "").strip()
CLOUDFLARE_API_TOKEN  = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()

@app.route("/test-provider", methods=["POST"])
def test_provider():
    data = request.json or {}
    provider = data.get("provider", "").lower().strip()
    
    test_message = [{"role": "user", "content": "Dis le mot 'Aligné'"}]
    gemini_test_message = "Dis le mot 'Aligné'"

    try:
        # 🟢 TEST GEMINI FREE
        if provider == "gemini_free":
            if not API_KEY_FREE: return jsonify({"status": "missing_key", "error": "API_KEY_FREE absente"}), 400
            client = genai.Client(api_key=API_KEY_FREE)
            r = client.models.generate_content(model="gemini-3.1-flash-lite", contents=gemini_test_message)
            return jsonify({"status": "success", "response": r.text.strip()})

        # 🔴 TEST GEMINI PAID
        elif provider == "gemini_paid":
            if not API_KEY_PAID: return jsonify({"status": "missing_key", "error": "API_KEY_PAID absente"}), 400
            client = genai.Client(api_key=API_KEY_PAID)
            r = client.models.generate_content(model="gemini-2.5-flash-lite", contents=gemini_test_message)
            return jsonify({"status": "success", "response": r.text.strip()})

        # 🟠 TEST OPENROUTER (ERNIE)
        elif provider == "openrouter":
            if not OPENROUTER_API_KEY: return jsonify({"status": "missing_key", "error": "OPENROUTER_API_KEY absente"}), 400
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
            res = client.chat.completions.create(model="baidu/ernie-4.5-vl-424b-a47b", messages=test_message, timeout=5.0)
            return jsonify({"status": "success", "response": res.choices[0].message.content.strip()})

        # 🔵 TEST GITHUB MODELS (DEEPSEEK)
        elif provider == "github":
            if not GITHUB_API_KEY: return jsonify({"status": "missing_key", "error": "GITHUB_API_KEY absente"}), 400
            client = OpenAI(base_url="https://models.github.ai/inference", api_key=GITHUB_API_KEY)
            res = client.chat.completions.create(model="deepseek/DeepSeek-V3-0324", messages=test_message, timeout=5.0)
            return jsonify({"status": "success", "response": res.choices[0].message.content.strip()})

        # 🟡 TEST NVIDIA (KIMI)
        elif provider == "nvidia":
            if not NVIDIA_API_KEY: return jsonify({"status": "missing_key", "error": "NVIDIA_API_KEY absente"}), 400
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
            res = client.chat.completions.create(model="moonshotai/kimi-k2.6", messages=test_message, timeout=5.0)
            return jsonify({"status": "success", "response": res.choices[0].message.content.strip()})

        # 🟣 TEST GROQ (COMPOUND)
        elif provider == "groq":
            if not GROQ_API_KEY: return jsonify({"status": "missing_key", "error": "GROQ_API_KEY absente"}), 400
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
            res = client.chat.completions.create(model="groq/compound", messages=test_message, timeout=5.0)
            return jsonify({"status": "success", "response": res.choices[0].message.content.strip()})

        # 🟤 TEST CLOUDFLARE (GLM)
        elif provider == "cloudflare":
            if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID: 
                return jsonify({"status": "missing_key", "error": "CLOUDFLARE_API_TOKEN ou ACCOUNT_ID absente"}), 400
            base_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
            client = OpenAI(base_url=base_url, api_key=CLOUDFLARE_API_TOKEN)
            res = client.chat.completions.create(model="@cf/zai-org/glm-4.7-flash", messages=test_message, timeout=5.0)
            return jsonify({"status": "success", "response": res.choices[0].message.content.strip()})

        else:
            return jsonify({"status": "error", "error": f"Provider '{provider}' inconnu."}), 400

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
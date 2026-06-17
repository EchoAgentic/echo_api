import os
import json
import re
import base64
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

# IMPORTATION DU FICHIER PROMPTS
from prompts import generate_system_prompt

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── CONFIGURATION DES CLÉS D'API ─────────────────────────────────────
API_KEY_PAID          = os.getenv("API_KEY_PAID", "").strip()
OPENROUTER_API_KEY    = os.getenv("OPENROUTER_API_KEY", "").strip()
GITHUB_API_KEY        = os.getenv("GITHUB_API_KEY", "").strip()
NVIDIA_API_KEY        = os.getenv("NVIDIA_API_KEY", "").strip()
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "").strip()
CLOUDFLARE_API_TOKEN  = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GITHUB_BASE_URL     = "https://models.github.ai/inference"
NVIDIA_BASE_URL     = "https://integrate.api.nvidia.com/v1"
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
CLOUDFLARE_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"

# ── INITIALISATION DES CLIENTS API ───────────────────────────────────
client_gemini_paid = genai.Client(api_key=API_KEY_PAID) if API_KEY_PAID else None
client_openrouter  = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY) if OPENROUTER_API_KEY else None
client_github      = OpenAI(base_url=GITHUB_BASE_URL,     api_key=GITHUB_API_KEY)     if GITHUB_API_KEY     else None
client_nvidia      = OpenAI(base_url=NVIDIA_BASE_URL,     api_key=NVIDIA_API_KEY)     if NVIDIA_API_KEY     else None
client_groq        = OpenAI(base_url=GROQ_BASE_URL,       api_key=GROQ_API_KEY)       if GROQ_API_KEY       else None
client_cloudflare  = OpenAI(base_url=CLOUDFLARE_BASE_URL, api_key=CLOUDFLARE_API_TOKEN) if CLOUDFLARE_API_TOKEN else None

# ── PARSAGE ET NETTOYAGE DU JSON DE RETOUR ────────────────────────────
def clean_and_parse_json(raw_text):
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "response" in parsed:
            return parsed
    except Exception:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except Exception:
            pass

    return {"action": None, "response": text}

# ── COMPILATION DU CONTEXTE CHATEXPERT POUR OPENROUTER ────────────────
def build_openrouter_messages(system_prompt, historique_reduit, user_message):
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in historique_reduit:
        if not isinstance(msg, str) or msg.startswith("__IMAGE__:"):
            continue
        clean_content = msg.split(":", 1)[1].strip() if ":" in msg else msg.strip()
        
        if msg.startswith("You:") or msg.startswith("Toi:"):
            messages.append({"role": "user", "content": clean_content})
        elif msg.startswith("Echo:"):
            messages.append({"role": "assistant", "content": clean_content})
            
    messages.append({"role": "user", "content": user_message})
    return messages

# ── COMPILATION DES CONTENUS DU CHAT POUR GEMINI ─────────────────────
def build_gemini_contents(historique_reduit: list, image_b64: str | None, user_message: str, force_neutral_style: bool) -> list:
    contents = []

    for msg in historique_reduit:
        if not isinstance(msg, str) or msg.startswith("__IMAGE__:"):
            continue
        clean_content = msg.split(":", 1)[1].strip() if ":" in msg else msg.strip()
        if "action limit reached" in clean_content.lower() or "do it for you" in clean_content.lower() or clean_content == "...":
            continue

        if msg.startswith("You:") or msg.startswith("Toi:"):
            contents.append({
                "role": "user",
                "parts": [types.Part.from_text(text=clean_content)]
            })
        elif msg.startswith("Echo:"):
            try:
                parsed = json.loads(clean_content)
                clean_content = parsed.get("response", clean_content)
            except Exception:
                pass
            
            if force_neutral_style:
                clean_content = "[Analyse technique archivée]"

            contents.append({
                "role": "model",
                "parts": [types.Part.from_text(text=clean_content)]
            })

    last_parts = []

    if image_b64:
        try:
            header, b64data = image_b64.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            raw_bytes = base64.b64decode(b64data)
            last_parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
        except Exception:
            pass

    text_to_send = user_message or "Analyse cette image."
    last_parts.append(types.Part.from_text(text=text_to_send))
    contents.append({"role": "user", "parts": last_parts})
    return contents

# ── ROUTE DE SANTÉ ───────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "online", "timestamp": datetime.now().isoformat()}), 200

# ── ROUTE PRINCIPALE /CHAT ───────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data             = request.json or {}
        user_message     = data.get("message", "")
        calendar_events  = data.get("calendarEvents", {})
        raw_history      = data.get("history", [])
        user_tier        = data.get("userTier", "free").lower().strip()
        source           = data.get("source", "chat").lower().strip()
        image_b64        = data.get("image", None)
        selected_buttons = data.get("selectedButtons", [])

        current_expenses = data.get("currentExpenses", [])
        current_calories = data.get("currentCalories", [])
        current_cycle    = data.get("currentCycle", "mois")

        maintenant       = datetime.now()
        date_aujourdhui  = maintenant.strftime("%A %d %B %Y")
        annee_en_cours   = maintenant.strftime("%Y")

        filtered_calendar = calendar_events

        # Génération du prompt système
        base_system_prompt = generate_system_prompt(
            source=source, selected_buttons=selected_buttons, date_aujourdhui=date_aujourdhui,
            annee_en_cours=annee_en_cours, user_tier=user_tier, filtered_calendar=filtered_calendar,
            current_expenses=current_expenses, current_calories=current_calories, current_cycle=current_cycle
        )
        system_prompt = base_system_prompt

        # Ajustement de la mémoire selon le forfait
        if user_tier == "free":
            taille_memoire = 5
            output_tokens = 1024
        elif user_tier in ["basic", "premium"]:
            taille_memoire = 15
            output_tokens = 2048
        else:
            taille_memoire = 30
            output_tokens = 4096

        historique_ajuste = raw_history[-taille_memoire:] if len(raw_history) > taille_memoire else raw_history

        # ── EXÉCUTION DU ROUTAGE SELON LE PLAN ───────────────────────────────
        
        # FORFAIT GRATUIT -> APPEL SÉCURISÉ OPENROUTER UNIQUE (GEMMA 4 FREE)
        if user_tier == "free":
            if not client_openrouter:
                return jsonify({"action": None, "response": "Client OpenRouter introuvable."}), 500
            try:
                print("[FREE] -> OpenRouter (google/gemma-4-26b-a4b:free)")
                or_messages = build_openrouter_messages(system_prompt, historique_ajuste, user_message)
                
                completion = client_openrouter.chat.completions.create(
                    model="google/gemma-4-26b-a4b:free",
                    messages=or_messages,
                    max_tokens=output_tokens
                )
                response_text = completion.choices[0].message.content
                return jsonify(clean_and_parse_json(response_text))
            except Exception as e:
                print(f"✕ Échec OpenRouter Free : {e}")
                return jsonify({"action": None, "response": "Mon sillage gratuit prend une pause, réessaie ! 😎"}), 500

        # FORFAITS PAYANTS -> APPELS DIRECTS GEMINI PAYANT
        else:
            if not client_gemini_paid:
                return jsonify({"action": None, "response": "Configuration d'API payante introuvable."}), 500

            has_active_buttons = len(selected_buttons) > 0
            gemini_contents = build_gemini_contents(
                historique_reduit=historique_ajuste, image_b64=image_b64, user_message=user_message,
                force_neutral_style=has_active_buttons or (source == "vitality")
            )

            def call_gemini(model_name):
                return client_gemini_paid.models.generate_content(
                    model=model_name, contents=gemini_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction, max_output_tokens=output_tokens
                    )
                )

            if user_tier in ["basic", "premium"]:
                try:
                    print("[PAID] -> Gemini 3.1 Flash-Lite")
                    r = call_gemini("gemini-3.1-flash-lite")
                    return jsonify(clean_and_parse_json(r.text))
                except Exception:
                    r = call_gemini("gemini-2.5-flash-lite")
                    return jsonify(clean_and_parse_json(r.text))

            elif user_tier == "ultra":
                try:
                    print("[PAID - ULTRA] -> Gemini 3.1 Flash-Lite")
                    r = call_gemini("gemini-3.1-flash-lite")
                    return jsonify(clean_and_parse_json(r.text))
                except Exception:
                    r = call_gemini("gemini-2.5-flash")
                    return jsonify(clean_and_parse_json(r.text))

            elif user_tier == "founder":
                try:
                    print("[PAID - FOUNDER] -> Gemini 3.0 Flash Preview")
                    r = call_gemini("gemini-3-flash-preview")
                    return jsonify(clean_and_parse_json(r.text))
                except Exception:
                    r = call_gemini("gemini-2.5-flash")
                    return jsonify(clean_and_parse_json(r.text))

    except Exception as e:
        return jsonify({"action": None, "response": f"Erreur critique: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
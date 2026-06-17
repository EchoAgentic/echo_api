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

from prompts import generate_system_prompt

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY_FREE          = os.getenv("API_KEY_FREE", "").strip()
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

client_gemini_free  = genai.Client(api_key=API_KEY_FREE)  if API_KEY_FREE  else None
client_gemini_paid  = genai.Client(api_key=API_KEY_PAID)  if API_KEY_PAID  else None
client_openrouter   = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY) if OPENROUTER_API_KEY else None
client_github       = OpenAI(base_url=GITHUB_BASE_URL,     api_key=GITHUB_API_KEY)     if GITHUB_API_KEY     else None
client_nvidia       = OpenAI(base_url=NVIDIA_BASE_URL,     api_key=NVIDIA_API_KEY)     if NVIDIA_API_KEY     else None
client_groq         = OpenAI(base_url=GROQ_BASE_URL,       api_key=GROQ_API_KEY)       if GROQ_API_KEY       else None
client_cloudflare   = OpenAI(base_url=CLOUDFLARE_BASE_URL, api_key=CLOUDFLARE_API_TOKEN) if CLOUDFLARE_API_TOKEN else None

GLOBAL_FAILOVER_MEMORY_COUNT = 0
MAX_FREE_FAILOVERS = 200

def get_failover_count():
    global GLOBAL_FAILOVER_MEMORY_COUNT
    return GLOBAL_FAILOVER_MEMORY_COUNT

def increment_failover_count():
    global GLOBAL_FAILOVER_MEMORY_COUNT
    GLOBAL_FAILOVER_MEMORY_COUNT += 1
    print(f"⚠️ [FAILOVER FREE] {GLOBAL_FAILOVER_MEMORY_COUNT} / {MAX_FREE_FAILOVERS}")

# ── NORMALISATION DU TIER ─────────────────────────────────────────────
VALID_TIERS = {"connected_free", "basic", "premium", "ultra", "founder"}

def normalize_tier(raw: str) -> str:
    cleaned = (raw or "").lower().strip()
    if cleaned in VALID_TIERS:
        return cleaned
    if cleaned == "free":
        return "connected_free"
    if "founder" in cleaned:
        return "founder"
    if "ultra" in cleaned:
        return "ultra"
    if "premium" in cleaned:
        return "premium"
    if "basic" in cleaned:
        return "basic"
    print(f"[WARN] Tier inconnu '{cleaned}' → connected_free")
    return "connected_free"

# ── JSON PARSER ───────────────────────────────────────────────────────
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

    match_array = re.search(r'\[\s*(\{.*?\})\s*,?', text, re.DOTALL)
    if match_array:
        try:
            parsed = json.loads(match_array.group(1))
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

    if text:
        clean_response = text
        if '"response":' in text:
            res_match = re.search(r'"response"\s*:\s*"([^"]+)"', text)
            if res_match:
                clean_response = res_match.group(1)
        return {"action": None, "response": clean_response}

    raise ValueError("Reponse vide.")

# ── BUILD GEMINI CONTENTS ─────────────────────────────────────────────
def build_gemini_contents(historique_reduit: list, image_b64: str | None, user_message: str, force_neutral_style: bool) -> list:
    contents = []

    for msg in historique_reduit:
        if not isinstance(msg, str) or msg.startswith("__IMAGE__:"):
            continue
        clean_content = msg.split(":", 1)[1].strip() if ":" in msg else msg.strip()
        if "action limit reached" in clean_content.lower() or "do it for you" in clean_content.lower() or clean_content == "...":
            continue

        if msg.startswith("You:") or msg.startswith("Toi:"):
            contents.append({"role": "user", "parts": [types.Part.from_text(text=clean_content)]})
        elif msg.startswith("Echo:"):
            try:
                parsed = json.loads(clean_content)
                clean_content = parsed.get("response", clean_content)
            except Exception:
                pass
            if force_neutral_style:
                clean_content = "[Analyse technique archivée]"
            contents.append({"role": "model", "parts": [types.Part.from_text(text=clean_content)]})

    last_parts = []

    if image_b64:
        try:
            header, b64data = image_b64.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            raw_bytes = base64.b64decode(b64data)
            last_parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
            print(f"[IMAGE] Injectée : {mime_type}, {len(raw_bytes)} bytes")
        except Exception as e:
            print(f"[WARN] Image decode error : {e}")

    text_to_send = user_message or "Analyse cette image et décris ce que tu vois."
    last_parts.append(types.Part.from_text(text=text_to_send))
    contents.append({"role": "user", "parts": last_parts})
    return contents

# ── HEALTH CHECK ──────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "message": "Le sillage d'Echo est parfaitement aligné.",
        "timestamp": datetime.now().isoformat()
    }), 200

# ── ROUTE PRINCIPALE ──────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data             = request.json or {}
        user_message     = data.get("message", "")
        calendar_events  = data.get("calendarEvents", {})
        raw_history      = data.get("history", [])
        source           = data.get("source", "chat").lower().strip()
        image_b64        = data.get("image", None)
        selected_buttons = data.get("selectedButtons", [])
        current_expenses = data.get("currentExpenses", [])
        current_calories = data.get("currentCalories", [])
        current_cycle    = data.get("currentCycle", "mois")

        # ── Normalisation robuste — "free" → "connected_free"
        user_tier = normalize_tier(data.get("userTier", "connected_free"))

        print(f"[DEBUG] message: '{user_message[:50]}' | tier: '{user_tier}' | source: '{source}' | image: {'oui' if image_b64 else 'non'}")

        maintenant      = datetime.now()
        date_aujourdhui = maintenant.strftime("%A %d %B %Y")
        annee_en_cours  = maintenant.strftime("%Y")

        filtered_calendar = {}
        date_debut = (maintenant - timedelta(days=31)).date()
        date_fin   = maintenant.date()
        try:
            for date_str, events in calendar_events.items():
                if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                    date_evt = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if date_debut <= date_evt <= date_fin:
                        filtered_calendar[date_str] = events
                else:
                    filtered_calendar[date_str] = events
        except Exception as e:
            print(f"[WARN] Erreur filtrage calendrier : {e}")
            filtered_calendar = calendar_events

        base_system_prompt = generate_system_prompt(
            source=source,
            selected_buttons=selected_buttons,
            date_aujourdhui=date_aujourdhui,
            annee_en_cours=annee_en_cours,
            user_tier=user_tier,
            filtered_calendar=filtered_calendar,
            current_expenses=current_expenses,
            current_calories=current_calories,
            current_cycle=current_cycle
        )

        anti_duplication_directive = (
            "\n\nCRITICAL SAFETY DIRECTIVE FOR ACTION EXTRACTION:\n"
            "You MUST ONLY trigger structural tools (calendar events, calories, expenses) "
            "if the intent is explicitly and newly demanded in the LATEST message. "
            "If the latest message is conversational, return 'action': null. "
            "Never repeat or re-execute a past action from historical text."
        )
        system_prompt = base_system_prompt + anti_duplication_directive

        # Taille mémoire & tokens selon tier
        if user_tier == "connected_free":
            taille_memoire = 8
            output_tokens  = 1024
        elif user_tier in ["basic", "premium"]:
            taille_memoire = 15
            output_tokens  = 2048
        else:  # ultra, founder
            taille_memoire = 30
            output_tokens  = 4096

        historique_ajuste = raw_history[-taille_memoire:] if len(raw_history) > taille_memoire else raw_history

        force_neutral = len(selected_buttons) > 0 or source == "vitality"
        gemini_contents = build_gemini_contents(historique_ajuste, image_b64, user_message, force_neutral)

        # ── call_gemini : PAS de response_mime_type, PAS de google_search tool
        # Ces deux options causent un 400 INVALID_ARGUMENT avec Gemini
        def call_gemini(client, model_name):
            return client.models.generate_content(
                model=model_name,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=output_tokens,
                )
            )

        # ── CONNECTED_FREE — tout sur clé payante, limites basses ─────
        if user_tier == "connected_free":
            if client_gemini_paid:
                for model in ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]:
                    try:
                        print(f"[CONNECTED_FREE] → {model}")
                        r = call_gemini(client_gemini_paid, model)
                        return jsonify(clean_and_parse_json(r.text))
                    except Exception as e:
                        print(f"[CONNECTED_FREE] Echec {model}: {e}")
            return jsonify({"action": None, "response": "Mon sillage rencontre un remous, réessaie ! 😎"})

        # ── BASIC / PREMIUM ───────────────────────────────────────────
        elif user_tier in ["basic", "premium"]:
            if client_gemini_paid:
                for model in ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]:
                    try:
                        print(f"[{user_tier.upper()}] → {model}")
                        r = call_gemini(client_gemini_paid, model)
                        return jsonify(clean_and_parse_json(r.text))
                    except Exception as e:
                        print(f"[{user_tier.upper()}] Echec {model}: {e}")
            return jsonify({"action": None, "response": "Petite friction, réessaie ! 😎"})

        # ── ULTRA ─────────────────────────────────────────────────────
        elif user_tier == "ultra":
            if client_gemini_paid:
                for model in ["gemini-3.1-flash-lite", "gemini-2.5-flash"]:
                    try:
                        print(f"[ULTRA] → {model}")
                        r = call_gemini(client_gemini_paid, model)
                        return jsonify(clean_and_parse_json(r.text))
                    except Exception as e:
                        print(f"[ULTRA] Echec {model}: {e}")
            return jsonify({"action": None, "response": "Mon sillage Ultra tangue, réessaie ! 😎"})

        # ── FOUNDER ───────────────────────────────────────────────────
        elif user_tier == "founder":
            if client_gemini_paid:
                for model in ["gemini-3-flash-preview", "gemini-2.5-flash"]:
                    try:
                        print(f"[FOUNDER] → {model}")
                        r = call_gemini(client_gemini_paid, model)
                        return jsonify(clean_and_parse_json(r.text))
                    except Exception as e:
                        print(f"[FOUNDER] Echec {model}: {e}")
            return jsonify({"action": None, "response": "Même les fondateurs ont des vagues. Réessaie ! 😎"})

        else:
            print(f"[WARN] Tier '{user_tier}' non géré — fallback connected_free")
            if client_gemini_paid:
                try:
                    r = call_gemini(client_gemini_paid, "gemini-2.5-flash-lite")
                    return jsonify(clean_and_parse_json(r.text))
                except Exception:
                    pass
            return jsonify({"action": None, "response": "Réessaie dans un instant ! 😎"})

    except Exception as e:
        import traceback
        print(f"[ERREUR CRITIQUE] {e}")
        traceback.print_exc()
        return jsonify({"action": None, "response": "Système instable, réessaie ton message !"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🔥 Serveur Echo connecté et aligné sur le port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
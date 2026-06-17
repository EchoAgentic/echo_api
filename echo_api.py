import os
import json
import re
import base64
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

from prompts import generate_system_prompt

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

API_KEY_PAID = os.getenv("API_KEY_PAID", "").strip()

client_gemini_paid = genai.Client(api_key=API_KEY_PAID) if API_KEY_PAID else None

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


def clean_and_parse_json(raw_text: str) -> dict:
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


def build_gemini_contents(historique: list, image_b64: str | None, user_message: str, force_neutral: bool) -> list:
    contents = []

    for msg in historique:
        if not isinstance(msg, str) or msg.startswith("__IMAGE__:"):
            continue
        clean = msg.split(":", 1)[1].strip() if ":" in msg else msg.strip()
        if "action limit reached" in clean.lower() or clean == "...":
            continue

        if msg.startswith("You:") or msg.startswith("Toi:"):
            contents.append({"role": "user", "parts": [types.Part.from_text(text=clean)]})
        elif msg.startswith("Echo:"):
            try:
                parsed = json.loads(clean)
                clean = parsed.get("response", clean)
            except Exception:
                pass
            if force_neutral:
                clean = "[Réponse archivée]"
            contents.append({"role": "model", "parts": [types.Part.from_text(text=clean)]})

    last_parts = []

    if image_b64:
        try:
            header, b64data = image_b64.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            raw_bytes = base64.b64decode(b64data)
            last_parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
        except Exception as e:
            print(f"[WARN] Image decode error: {e}")

    last_parts.append(types.Part.from_text(text=user_message or "Analyse cette image."))
    contents.append({"role": "user", "parts": last_parts})
    return contents


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "online", "timestamp": datetime.now().isoformat()}), 200


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}

        user_message     = data.get("message", "")
        calendar_events  = data.get("calendarEvents", {})
        raw_history      = data.get("history", [])
        source           = data.get("source", "chat").lower().strip()
        image_b64        = data.get("image", None)
        selected_buttons = data.get("selectedButtons", [])
        current_expenses = data.get("currentExpenses", [])
        current_calories = data.get("currentCalories", [])
        current_cycle    = data.get("currentCycle", "mois")

        # ── Normalisation robuste — "free" → "connected_free", tout inconnu → "connected_free"
        user_tier = normalize_tier(data.get("userTier", "connected_free"))

        maintenant      = datetime.now()
        date_aujourdhui = maintenant.strftime("%A %d %B %Y")
        annee_en_cours  = maintenant.strftime("%Y")

        print(f"[DEBUG] message: '{user_message[:50]}' | tier: '{user_tier}' | source: '{source}' | image: {'oui' if image_b64 else 'non'}")

        system_prompt = generate_system_prompt(
            source=source,
            selected_buttons=selected_buttons,
            date_aujourdhui=date_aujourdhui,
            annee_en_cours=annee_en_cours,
            user_tier=user_tier,
            filtered_calendar=calendar_events,
            current_expenses=current_expenses,
            current_calories=current_calories,
            current_cycle=current_cycle,
        )

        # Taille mémoire & tokens
        if user_tier == "connected_free":
            taille_memoire = 8
            output_tokens  = 1024
        elif user_tier in ["basic", "premium"]:
            taille_memoire = 15
            output_tokens  = 2048
        else:
            taille_memoire = 30
            output_tokens  = 4096

        historique = raw_history[-taille_memoire:] if len(raw_history) > taille_memoire else raw_history

        if not client_gemini_paid:
            return jsonify({"action": None, "response": "Configuration API introuvable."}), 500

        force_neutral = len(selected_buttons) > 0 or source == "vitality"
        gemini_contents = build_gemini_contents(historique, image_b64, user_message, force_neutral)

        def call_gemini(model_name: str):
            # ── PAS de response_mime_type ici — incompatible avec le mode multimodal Gemini
            return client_gemini_paid.models.generate_content(
                model=model_name,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=output_tokens,
                ),
            )

        # ── CONNECTED_FREE ────────────────────────────────────────────────────
        if user_tier == "connected_free":
            for model in ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]:
                try:
                    print(f"[CONNECTED_FREE] → {model}")
                    r = call_gemini(model)
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"[CONNECTED_FREE] Echec {model}: {e}")
            return jsonify({"action": None, "response": "Mon sillage rencontre un remous, réessaie ! 😎"})

        # ── BASIC / PREMIUM ───────────────────────────────────────────────────
        elif user_tier in ["basic", "premium"]:
            for model in ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]:
                try:
                    print(f"[{user_tier.upper()}] → {model}")
                    r = call_gemini(model)
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"[{user_tier.upper()}] Echec {model}: {e}")
            return jsonify({"action": None, "response": "Petite friction, réessaie ! 😎"})

        # ── ULTRA ─────────────────────────────────────────────────────────────
        elif user_tier == "ultra":
            for model in ["gemini-3.1-flash-lite", "gemini-2.5-flash"]:
                try:
                    print(f"[ULTRA] → {model}")
                    r = call_gemini(model)
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"[ULTRA] Echec {model}: {e}")
            return jsonify({"action": None, "response": "Mon sillage Ultra tangue, réessaie ! 😎"})

        # ── FOUNDER ───────────────────────────────────────────────────────────
        elif user_tier == "founder":
            for model in ["gemini-3-flash-preview", "gemini-2.5-flash"]:
                try:
                    print(f"[FOUNDER] → {model}")
                    r = call_gemini(model)
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"[FOUNDER] Echec {model}: {e}")
            return jsonify({"action": None, "response": "Même les fondateurs ont des vagues. Réessaie ! 😎"})

        # ── FALLBACK (ne devrait jamais arriver) ──────────────────────────────
        else:
            print(f"[WARN] Tier '{user_tier}' non géré après normalisation")
            try:
                r = call_gemini("gemini-2.5-flash-lite")
                return jsonify(clean_and_parse_json(r.text))
            except Exception as e:
                return jsonify({"action": None, "response": "Réessaie dans un instant ! 😎"})

    except Exception as e:
        print(f"[ERREUR CRITIQUE] {e}")
        return jsonify({"action": None, "response": f"Erreur critique : {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🔥 Serveur Echo démarré sur le port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

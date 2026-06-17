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
CORS(app)

# ── CONFIGURATION DES CLÉS D'API ─────────────────────────────────────
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

# ── INITIALISATION DES CLIENTS API ───────────────────────────────────
client_gemini_free  = genai.Client(api_key=API_KEY_FREE)  if API_KEY_FREE  else None
client_gemini_paid  = genai.Client(api_key=API_KEY_PAID)  if API_KEY_PAID  else None
client_openrouter   = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY) if OPENROUTER_API_KEY else None
client_github       = OpenAI(base_url=GITHUB_BASE_URL,     api_key=GITHUB_API_KEY)     if GITHUB_API_KEY     else None
client_nvidia       = OpenAI(base_url=NVIDIA_BASE_URL,     api_key=NVIDIA_API_KEY)     if NVIDIA_API_KEY     else None
client_groq         = OpenAI(base_url=GROQ_BASE_URL,       api_key=GROQ_API_KEY)       if GROQ_API_KEY       else None
client_cloudflare   = OpenAI(base_url=CLOUDFLARE_BASE_URL, api_key=CLOUDFLARE_API_TOKEN) if CLOUDFLARE_API_TOKEN else None

# SÉCURISATION DES LIMITES DE FAILOVER VIA VARIABLE DE MÉMOIRE RUNTIME GLOBAL
GLOBAL_FAILOVER_MEMORY_COUNT = 0
MAX_FREE_FAILOVERS = 200

def get_failover_count():
    global GLOBAL_FAILOVER_MEMORY_COUNT
    return GLOBAL_FAILOVER_MEMORY_COUNT

def increment_failover_count():
    global GLOBAL_FAILOVER_MEMORY_COUNT
    GLOBAL_FAILOVER_MEMORY_COUNT += 1
    print(f"⚠️ [FAILOVER FREE MEMORY] {GLOBAL_FAILOVER_MEMORY_COUNT} / {MAX_FREE_FAILOVERS}")

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
            print(f"[IMAGE] Injectée : {mime_type}, {len(raw_bytes)} bytes")
        except Exception as e:
            print(f"[WARN] Image decode error : {e} — envoi sans image")

    text_to_send = user_message or "Analyse cette image et décris ce que tu vois."
    last_parts.append(types.Part.from_text(text=text_to_send))

    contents.append({
        "role": "user",
        "parts": last_parts
    })

    return contents

# ── ROUTE DE SANTÉ (CRITIQUE POUR LES DÉPLOIEMENTS RAILWAY/RENDER) ────
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "message": "Le sillage d'Echo est parfaitement aligné sur son Axe en production.",
        "presence": "Echo",
        "timestamp": datetime.now().isoformat()
    }), 200

# ── ROUTE PRINCIPALE DE DISCUSSION /CHAT ─────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data             = request.json or {}
        user_message     = data.get("message", "")
        calendar_events = data.get("calendarEvents", {})
        raw_history      = data.get("history", [])
        user_tier        = data.get("userTier", "free").lower().strip()
        source           = data.get("source", "chat").lower().strip()
        image_b64        = data.get("image", None)
        selected_buttons = data.get("selectedButtons", [])

        # Extraction des paramètres de l'outil Vitalité
        current_expenses = data.get("currentExpenses", [])
        current_calories = data.get("currentCalories", [])
        current_cycle    = data.get("currentCycle", "mois")

        has_image = bool(image_b64)
        print(f"[DEBUG] message: '{user_message[:60]}' | tier: '{user_tier}' | source: '{source}' | image: {has_image} | boutons: {selected_buttons}")

        maintenant       = datetime.now()
        date_aujourdhui  = maintenant.strftime("%A %d %B %Y")
        annee_en_cours   = maintenant.strftime("%Y")

        # Filtrage des événements du calendrier (fenêtre de 31 jours)
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

        # ── APPEL DE LA LOGIQUE EXTERNALISÉE DU PROMPT SYSTÈME ─────────
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

        # ── 🛠️ DIRECTION CRITIQUE ANTI-DOUBLONS (CHANTIER 1) ──
        anti_duplication_directive = (
            "\n\nCRITICAL SAFETY DIRECTIVE FOR ACTION EXTRACTION:\n"
            "You are provided with a partial chat history for baseline conversational context. However, you MUST ONLY "
            "trigger or extract updates for structural tools (like adding calendar events, logging calories, or recording expenses) "
            "if the intent is explicitly and newly demanded in the LATEST message from the user. "
            "If the latest user message is conversational, neutral, or just a reply to your previous text, you MUST return "
            "'action': null, even if earlier blocks of the chat history contain command sentences. "
            "Never repeat or re-execute a past action from historical text."
        )
        system_prompt = base_system_prompt + anti_duplication_directive

        # =====================================================================
        # AJUSTEMENT DYNAMIQUE DE LA MÉMOIRE ET DES TOKENS SELON LE PLAN
        # =====================================================================
        if user_tier == "free":
            taille_memoire = 5
            output_tokens = 1024
        elif user_tier in ["basic", "premium"]:
            taille_memoire = 15
            output_tokens = 2048
        elif user_tier in ["ultra", "founder"]:
            taille_memoire = 30
            output_tokens = 4096
        else:
            taille_memoire = 5
            output_tokens = 1024

        # Extraction fine des derniers messages selon le forfait de l'utilisateur
        historique_ajuste = raw_history[-taille_memoire:] if len(raw_history) > taille_memoire else raw_history

        has_active_buttons = len(selected_buttons) > 0
        gemini_contents = build_gemini_contents(
            historique_reduit=historique_ajuste, 
            image_b64=image_b64, 
            user_message=user_message, 
            force_neutral_style=has_active_buttons or (source == "vitality")
        )

        # ── INJECTION DE GOOGLE SEARCH RETRIEVAL (CORRIGÉ ET SÉCURISÉ) ───────
        def call_gemini(client, model_name):
            # Retrait du response_mime_type strict incompatible avec Flash-Lite + Google Search
            return client.models.generate_content(
                model=model_name,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=output_tokens,
                    tools=[{"google_search": {}}]
                )
            )

        # =====================================================================
        # ROUTAGE ET STRATÉGIE DE FAILOVER SECURISÉE (SOUVERAINETÉ DE COÛTS)
        # =====================================================================
        
        # ── FORFAIT GRATUIT (FREE) ───────────────────────────────────────────
        if user_tier == "free":
            current_failovers = get_failover_count()
            if current_failovers >= MAX_FREE_FAILOVERS:
                print(f"🔒 [CUT-CIRCUIT EXCEEDED] Coupe-circuit activé ({current_failovers}/{MAX_FREE_FAILOVERS})")
                return jsonify({"action": None, "response": "Ouf, mon sillage sature sous l'affluence gratuite. Laisse-moi une petite seconde ! 😎"})

            if client_gemini_free:
                try:
                    print("[FREE 1/3] -> Gemini 3.1 Flash-Lite")
                    r = call_gemini(client_gemini_free, "gemini-3.1-flash-lite")
                    if r and hasattr(r, 'text') and r.text:
                        return jsonify(clean_and_parse_json(r.text))
                    else:
                        print("✕ Gemini 3.1 Free a renvoyé une réponse vide.")
                except Exception as e:
                    print(f"Echec Gemini 3.1 Free ({e})")

                try:
                    print("[FREE 2/3] -> Gemini 2.5 Flash-Lite")
                    r = call_gemini(client_gemini_free, "gemini-2.5-flash-lite")
                    if r and hasattr(r, 'text') and r.text:
                        return jsonify(clean_and_parse_json(r.text))
                    else:
                        print("✕ Gemini 2.5 Free a renvoyé une réponse vide.")
                except Exception as e:
                    print(f"Echec Gemini 2.5 Free ({e})")

            # Filet de secours payant pour les gratuits (sécurisé en mémoire vive)
            if client_gemini_paid:
                try:
                    print(f"🚨 [FREE -> FILET] Gemini 2.5 Flash-Lite Payant ({current_failovers + 1}/{MAX_FREE_FAILOVERS})")
                    r = call_gemini(client_gemini_paid, "gemini-2.5-flash-lite")
                    if r and hasattr(r, 'text') and r.text:
                        increment_failover_count()
                        return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"✕ Échec filet de sécurité : {e}")

            return jsonify({"action": None, "response": "Ouf, mon sillage sature sous l'affluence gratuite. Laisse-moi une petite seconde ! 😎"})

        # ── FORFAIT INTERMÉDIAIRE (BASIC / PREMIUM) ──────────────────────────
        elif user_tier in ["basic", "premium"]:
            if client_gemini_paid:
                try:
                    print("[PAID] -> Gemini 2.5 Flash-Lite Premium")
                    r = call_gemini(client_gemini_paid, "gemini-2.5-flash-lite")
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"⚠️ [FAILOVER BASIC/PREMIUM] Gemini 2.5 Flash-Lite saturé ({e}). Bascule sur 3.1 Flash-Lite...")
                    try:
                        r = call_gemini(client_gemini_paid, "gemini-3.1-flash-lite")
                        return jsonify(clean_and_parse_json(r.text))
                    except Exception as fallback_error:
                        print(f"✕ Échec critique du failover Basic/Premium : {fallback_error}")
                        return jsonify({"action": None, "response": "Je ressens une petite déconnexion, redonne-moi une seconde ! 😎"})
            return jsonify({"action": None, "response": "Configuration d'API payante introuvable."}), 500

        # ── FORFAIT HAUTE PERFORMANCE (ULTRA) ───────────────────────────────
        elif user_tier == "ultra":
            if client_gemini_paid:
                try:
                    print("[PAID - ULTRA] -> Gemini 3.1 Flash-Lite Payant")
                    r = call_gemini(client_gemini_paid, "gemini-3.1-flash-lite")
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"⚠️ [FAILOVER ULTRA] Gemini 3.1 Lite saturé ({e}). Bascule de puissance sur 2.5 Flash complet...")
                    try:
                        r = call_gemini(client_gemini_paid, "gemini-2.5-flash")
                        return jsonify(clean_and_parse_json(r.text))
                    except Exception as fallback_error:
                        print(f"✕ Échec critique du failover Ultra : {fallback_error}")
                        return jsonify({"action": None, "response": "Mon sillage Ultra tangue un peu. Réessaie ! 😎"})
            return jsonify({"action": None, "response": "Configuration d'API payante introuvable."}), 500

        # ── FORFAIT ÉLITE / HISTORIQUE (FOUNDER) ────────────────────────────
        elif user_tier == "founder":
            if client_gemini_paid:
                try:
                    print("[PAID - FOUNDER] -> Gemini 3.0 Flash Preview")
                    r = call_gemini(client_gemini_paid, "gemini-3-flash-preview")
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"⚠️ [FAILOVER FOUNDER] Erreur sur Gemini 3.0 ({e}). Bascule immédiate sur 2.5 Flash...")
                    try:
                        r = call_gemini(client_gemini_paid, "gemini-2.5-flash")
                        return jsonify(clean_and_parse_json(r.text))
                    except Exception as fallback_error:
                        print(f"✕ Échec critique du failover Founder : {fallback_error}")
                        return jsonify({"action": None, "response": "Même les fondateurs ont parfois des vagues. Réessaie ! 😎"})
            return jsonify({"action": None, "response": "Configuration d'API payante introuvable."}), 500

        else:
            return jsonify({"action": None, "response": "Plan utilisateur non reconnu."}), 400

    except Exception as e:
        print(f"Erreur critique Flask : {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"action": None, "response": "Système instable, réessaie ton message !"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🔥 Serveur Echo connecté et aligné sur le port {port}...")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
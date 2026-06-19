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

# ── CLÉS D'API ────────────────────────────────────────────────────────────────
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

# ── CLIENTS ───────────────────────────────────────────────────────────────────
client_gemini_free  = genai.Client(api_key=API_KEY_FREE)  if API_KEY_FREE  else None
client_gemini_paid  = genai.Client(api_key=API_KEY_PAID)  if API_KEY_PAID  else None
client_openrouter   = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY) if OPENROUTER_API_KEY else None
client_github       = OpenAI(base_url=GITHUB_BASE_URL,     api_key=GITHUB_API_KEY)     if GITHUB_API_KEY     else None
client_nvidia       = OpenAI(base_url=NVIDIA_BASE_URL,     api_key=NVIDIA_API_KEY)     if NVIDIA_API_KEY     else None
client_groq         = OpenAI(base_url=GROQ_BASE_URL,        api_key=GROQ_API_KEY)        if GROQ_API_KEY       else None
client_cloudflare   = OpenAI(base_url=CLOUDFLARE_BASE_URL, api_key=CLOUDFLARE_API_TOKEN) if CLOUDFLARE_API_TOKEN else None

# ── FILET DE SÉCURITÉ CONNECTED_FREE ─────────────────────────────────────────
GLOBAL_FAILOVER_MEMORY_COUNT = 0
MAX_FREE_FAILOVERS = 200

def get_failover_count():
    global GLOBAL_FAILOVER_MEMORY_COUNT
    return GLOBAL_FAILOVER_MEMORY_COUNT

def increment_failover_count():
    global GLOBAL_FAILOVER_MEMORY_COUNT
    GLOBAL_FAILOVER_MEMORY_COUNT += 1
    print(f"⚠️ [FAILOVER FILET] {GLOBAL_FAILOVER_MEMORY_COUNT} / {MAX_FREE_FAILOVERS}")

# ── GESTION DES LOCKS DE MODÈLES (60 SECONDES) ────────────────────────────────
MODELS_LOCK_REGISTRY = {}

def is_model_locked(model_name: str) -> bool:
    """Vérifie si un modèle est bloqué suite à un échec récent."""
    lock_time = MODELS_LOCK_REGISTRY.get(model_name)
    if lock_time:
        if datetime.now() < lock_time:
            print(f"🔒 [CIRCUIT BREAKER] {model_name} est bloqué pour encore {(lock_time - datetime.now()).total_seconds():.1f}s")
            return True
        else:
            del MODELS_LOCK_REGISTRY[model_name]
    return False

def lock_model(model_name: str):
    """Bloque un modèle pendant 60 secondes suite à une erreur."""
    MODELS_LOCK_REGISTRY[model_name] = datetime.now() + timedelta(seconds=60)
    print(f"🚨 [LOCK ACTIVE] {model_name} est mis hors circuit pendant 60 secondes.")

# ── NORMALISATION DU TIER ─────────────────────────────────────────────────────
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

# ── JSON PARSER ───────────────────────────────────────────────────────────────
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

# ── BUILD GEMINI CONTENTS ─────────────────────────────────────────────────────
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
            print(f"[WARN] Image decode error : {e} — envoi sans image")

    text_to_send = user_message or "Analyse cette image et décris ce que tu vois."
    last_parts.append(types.Part.from_text(text=text_to_send))
    contents.append({"role": "user", "parts": last_parts})
    return contents

# ── BUILD SYSTEM PROMPT ET PAYLOAD OPENROUTER CONTEXT ─────────────────────────
def prepare_shared_context(data):
    user_message     = data.get("message", "")
    calendar_events  = data.get("calendarEvents", {})
    raw_history      = data.get("history", [])
    source           = data.get("source", "chat").lower().strip()
    image_b64        = data.get("image", None)
    selected_buttons = data.get("selectedButtons", [])
    current_expenses = data.get("currentExpenses", [])
    current_calories = data.get("currentCalories", [])
    current_cycle    = data.get("currentCycle", "mois")
    user_tier        = normalize_tier(data.get("userTier", "connected_free"))

    maintenant = datetime.now()
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
        source=source, selected_buttons=selected_buttons, date_aujourdhui=date_aujourdhui,
        annee_en_cours=annee_en_cours, user_tier=user_tier, filtered_calendar=filtered_calendar,
        current_expenses=current_expenses, current_calories=current_calories, current_cycle=current_cycle
    )

    anti_duplication_directive = (
        "\n\nCRITICAL SAFETY DIRECTIVE FOR ACTION EXTRACTION:\n"
        "You must ONLY trigger updates for structural tools if explicitly and newly demanded in the LATEST message.\n"
        "Never repeat or re-execute a past action from historical text."
    )
    system_prompt = base_system_prompt + anti_duplication_directive

    taille_memoire = 30 if user_tier in ["ultra", "founder"] else (15 if user_tier in ["basic", "premium"] else 5)
    output_tokens  = 4096 if user_tier in ["ultra", "founder"] else (2048 if user_tier in ["basic", "premium"] else 1024)
    
    historique_ajuste = raw_history[-taille_memoire:] if len(raw_history) > taille_memoire else raw_history
    force_neutral = len(selected_buttons) > 0 or source == "vitality"
    
    gemini_contents = build_gemini_contents(historique_ajuste, image_b64, user_message, force_neutral)

    messages_openrouter = [{"role": "system", "content": system_prompt}]
    for msg in historique_ajuste:
        if not isinstance(msg, str) or msg.startswith("__IMAGE__:"):
            continue
        clean_content = msg.split(":", 1)[1].strip() if ":" in msg else msg.strip()
        if "action limit reached" in clean_content.lower() or "do it for you" in clean_content.lower() or clean_content == "...":
            continue
        if msg.startswith("You:") or msg.startswith("Toi:"):
            messages_openrouter.append({"role": "user", "content": clean_content})
        elif msg.startswith("Echo:"):
            try:
                parsed = json.loads(clean_content)
                clean_content = parsed.get("response", clean_content)
            except Exception:
                pass
            messages_openrouter.append({"role": "assistant", "content": clean_content})

    if user_message:
        messages_openrouter.append({"role": "user", "content": clean_content})

    return {
        "system_prompt": system_prompt,
        "output_tokens": output_tokens,
        "gemini_contents": gemini_contents,
        "messages_openrouter": messages_openrouter,
        "user_tier": user_tier
    }

# ── LOGIQUE COMMUNE D'APPEL DE CAPTURE ────────────────────────────────────────
def execute_gemini_call(client, model, ctx):
    return client.models.generate_content(
        model=model, contents=ctx["gemini_contents"],
        config=types.GenerateContentConfig(system_instruction=ctx["system_prompt"], max_output_tokens=ctx["output_tokens"])
    )

def execute_openai_call(client, model, ctx, temp=0.7, timeout=7.0):
    res = client.chat.completions.create(
        model=model, messages=ctx["messages_openrouter"], temperature=temp, timeout=timeout
    )
    return res.choices[0].message.content

# ── ROUTE CLAVARDAGE (CHAT TRADITIONNEL) ──────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        ctx = prepare_shared_context(data)

        if ctx["user_tier"] == "connected_free":
            current_failovers = get_failover_count()
            if current_failovers >= MAX_FREE_FAILOVERS:
                return jsonify({"action": None, "response": "Ouf, mon sillage sature sous l'affluence. Laisse-moi une petite seconde ! 😎"})

            # 1/3 — Gemini 3.1 Flash-Lite
            model_1 = "gemini-3.1-flash-lite"
            if client_gemini_free and not is_model_locked(model_1):
                try:
                    print("[CHAT - FREE 1/3] → Gemini 3.1 Flash-Lite (FREE KEY)")
                    r = execute_gemini_call(client_gemini_free, model_1, ctx)
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"Échec {model_1} ({e})")
                    lock_model(model_1)

            # 2/3 — Gemini 2.5 Flash-Lite
            model_2 = "gemini-2.5-flash-lite"
            if client_gemini_free and not is_model_locked(model_2):
                try:
                    print("[CHAT - FREE 2/3] → Gemini 2.5 Flash-Lite (FREE KEY)")
                    r = execute_gemini_call(client_gemini_free, model_2, ctx)
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"Échec {model_2} ({e})")
                    lock_model(model_2)

            # 3/3 — Filet Payant
            if client_gemini_paid and not is_model_locked("gemini-2.5-flash-lite-paid"):
                try:
                    print(f"[CHAT - FREE 3/3 FILET] → Gemini 2.5 Flash-Lite (PAID KEY) ({current_failovers + 1}/{MAX_FREE_FAILOVERS})")
                    r = execute_gemini_call(client_gemini_paid, "gemini-2.5-flash-lite", ctx)
                    increment_failover_count()
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"Échec Filet Payant ({e})")
                    lock_model("gemini-2.5-flash-lite-paid")

            return jsonify({"action": None, "response": "Ouf, mon sillage sature. Laisse-moi une petite seconde ! 😎"})

        # PLANS PREMIUM / ULTRA / FOUNDER
        else:
            target_model = "gemini-3.5-flash" if ctx["user_tier"] == "founder" else "gemini-3.1-flash-lite"
            try:
                print(f"[CHAT - PAID] → {target_model}")
                r = execute_gemini_call(client_gemini_paid, target_model, ctx)
                return jsonify(clean_and_parse_json(r.text))
            except Exception as e:
                print(f"Échec {target_model} Premium ({e})")
                # Fallback universel payant de secours ultime
                r = execute_gemini_call(client_gemini_paid, "gemini-2.5-flash-lite", ctx)
                return jsonify(clean_and_parse_json(r.text))

    except Exception as e:
        print(f"Erreur critique /chat : {e}")
        return jsonify({"action": None, "response": "Système instable, réessaie ton message !"}), 500

# ── ROUTE BOOKS (ÉCRITURE STUDIO) ────────────────────────────────────────────
@app.route("/books", methods=["POST"])
def books():
    try:
        data = request.json or {}
        ctx = prepare_shared_context(data)

        if ctx["user_tier"] == "connected_free":
            current_failovers = get_failover_count()
            if current_failovers >= MAX_FREE_FAILOVERS:
                return jsonify({"action": None, "response": "Mon sillage d'écriture est saturé pour l'instant ! 😎"})

            # 1/3 — Ernie 4.5 VL (OpenRouter)
            model_1 = "baidu/ernie-4.5-vl-424b-a47b"
            if client_openrouter and not is_model_locked(model_1):
                try:
                    print("[BOOKS - FREE 1/3] → Ernie 4.5 (OpenRouter)")
                    res = execute_openai_call(client_openrouter, model_1, ctx)
                    return jsonify(clean_and_parse_json(res))
                except Exception as e:
                    print(f"Échec {model_1} ({e})")
                    lock_model(model_1)

            # 2/3 — GLM 4.7 Flash (Cloudflare)
            model_2 = "@cf/zai-org/glm-4.7-flash"
            if client_cloudflare and not is_model_locked(model_2):
                try:
                    print("[BOOKS - FREE 2/3] → GLM 4.7 Flash (Cloudflare)")
                    res = execute_openai_call(client_cloudflare, model_2, ctx)
                    return jsonify(clean_and_parse_json(res))
                except Exception as e:
                    print(f"Échec {model_2} ({e})")
                    lock_model(model_2)

            # 3/3 — Gemini 2.5 Flash-Lite Paid
            if client_gemini_paid and not is_model_locked("gemini-2.5-flash-lite-books"):
                try:
                    print(f"[BOOKS - FREE 3/3 FILET] → Gemini 2.5 Flash-Lite (PAID KEY)")
                    r = execute_gemini_call(client_gemini_paid, "gemini-2.5-flash-lite", ctx)
                    increment_failover_count()
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"Échec Filet Payant Books ({e})")
                    lock_model("gemini-2.5-flash-lite-books")

        return jsonify({"action": None, "response": "Ouf, mon sillage d'écriture sature. Laisse-moi une seconde ! 😎"})
    except Exception as e:
        print(f"Erreur critique /books : {e}")
        return jsonify({"action": None, "response": "Studio d'écriture instable, réessaie !"}), 500

# ── ROUTE HOME (TABLEAU DE BORD ACCUEIL) ──────────────────────────────────────
@app.route("/home", methods=["POST"])
def home():
    try:
        data = request.json or {}
        ctx = prepare_shared_context(data)

        if ctx["user_tier"] == "connected_free":
            current_failovers = get_failover_count()
            if current_failovers >= MAX_FREE_FAILOVERS:
                return jsonify({"action": None, "response": "Mon sillage d'accueil est au repos forcé ! 😎"})

            # 1/3 — DeepSeek V3 (GitHub)
            model_1 = "deepseek/DeepSeek-V3-0324"
            if client_github and not is_model_locked(model_1):
                try:
                    print("[HOME - FREE 1/3] → DeepSeek V3 (GitHub)")
                    res = execute_openai_call(client_github, model_1, ctx)
                    return jsonify(clean_and_parse_json(res))
                except Exception as e:
                    print(f"Échec {model_1} ({e})")
                    lock_model(model_1)

            # 2/3 — Kimi K2.6 (NVIDIA)
            model_2 = "moonshotai/kimi-k2.6"
            if client_nvidia and not is_model_locked(model_2):
                try:
                    print("[HOME - FREE 2/3] → Kimi K2.6 (NVIDIA)")
                    res = execute_openai_call(client_nvidia, model_2, ctx)
                    return jsonify(clean_and_parse_json(res))
                except Exception as e:
                    print(f"Échec {model_2} ({e})")
                    lock_model(model_2)

            # 3/3 — Gemini 2.5 Flash-Lite Paid
            if client_gemini_paid and not is_model_locked("gemini-2.5-flash-lite-home"):
                try:
                    print(f"[HOME - FREE 3/3 FILET] → Gemini 2.5 Flash-Lite (PAID KEY)")
                    r = execute_gemini_call(client_gemini_paid, "gemini-2.5-flash-lite", ctx)
                    increment_failover_count()
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"Échec Filet Payant Home ({e})")
                    lock_model("gemini-2.5-flash-lite-home")

        return jsonify({"action": None, "response": "Espace d'accueil surchargé, redonne-moi une seconde ! 😎"})
    except Exception as e:
        print(f"Erreur critique /home : {e}")
        return jsonify({"action": None, "response": "Espace d'accueil instable, réessaie !"}), 500

# ── ROUTE HISTORY (ARCHIVES & SILLAGES PASSÉS) ────────────────────────────────
@app.route("/history", methods=["POST"])
def history():
    try:
        data = request.json or {}
        ctx = prepare_shared_context(data)

        if ctx["user_tier"] == "connected_free":
            current_failovers = get_failover_count()
            if current_failovers >= MAX_FREE_FAILOVERS:
                return jsonify({"action": None, "response": "Les archives historiques sont momentanément inaccessibles ! 😎"})

            # 1/3 — Groq Compound (Groq)
            model_1 = "groq/compound"
            if client_groq and not is_model_locked(model_1):
                try:
                    print("[HISTORY - FREE 1/3] → Groq Compound")
                    res = execute_openai_call(client_groq, model_1, ctx)
                    return jsonify(clean_and_parse_json(res))
                except Exception as e:
                    print(f"Échec {model_1} ({e})")
                    lock_model(model_1)

            # 2/3 — Kimi K2.6 (NVIDIA)
            model_2 = "moonshotai/kimi-k2.6"
            if client_nvidia and not is_model_locked(model_2):
                try:
                    print("[HISTORY - FREE 2/3] → Kimi K2.6 (NVIDIA)")
                    res = execute_openai_call(client_nvidia, model_2, ctx)
                    return jsonify(clean_and_parse_json(res))
                except Exception as e:
                    print(f"Échec {model_2} ({e})")
                    lock_model(model_2)

            # 3/3 — Gemini 2.5 Flash-Lite Paid
            if client_gemini_paid and not is_model_locked("gemini-2.5-flash-lite-history"):
                try:
                    print(f"[HISTORY - FREE 3/3 FILET] → Gemini 2.5 Flash-Lite (PAID KEY)")
                    r = execute_gemini_call(client_gemini_paid, "gemini-2.5-flash-lite", ctx)
                    increment_failover_count()
                    return jsonify(clean_and_parse_json(r.text))
                except Exception as e:
                    print(f"Échec Filet Payant History ({e})")
                    lock_model("gemini-2.5-flash-lite-history")

        return jsonify({"action": None, "response": "Module d'historique saturé, redonne-moi une seconde ! 😎"})
    except Exception as e:
        print(f"Erreur critique /history : {e}")
        return jsonify({"action": None, "response": "Historique instable, réessaie !"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🔥 Serveur Multi-Cascades branché et synchronisé sur le port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
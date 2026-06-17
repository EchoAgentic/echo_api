import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Charge le fichier .env local pour choper la clé API_KEY_PAID
load_dotenv()

API_KEY_PAID = os.getenv("API_KEY_PAID", "").strip()

print("📡 [TEST LOCAL] Initialisation du client...")
print(f"🔑 Clé API détectée : {API_KEY_PAID[:6]}...{API_KEY_PAID[-4:] if len(API_KEY_PAID) > 4 else ''}\n")

if not API_KEY_PAID:
    print("❌ [ERREUR] API_KEY_PAID est introuvable. Vérifie ton fichier .env dans ce dossier !")
    exit()

try:
    # On initialise le client Google GenAI
    client = genai.Client(api_key=API_KEY_PAID)
    
    print("🔄 [APPEL LOCAL] Envoi de la requête à gemini-3.1-flash-lite...")
    
    # Config ultra-stable (Recherche Web active, pas de JSON strict pour pas bloquer)
    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        tools=[{"google_search": {}}]
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents="Dis-moi bonjour et confirme que ton sillage local répond à 100% sur le modèle 3.1 !",
        config=config
    )

    if response and hasattr(response, 'text') and response.text:
        print("\n✅ [SUCCÈS LOCAL] Réponse d'Echo :")
        print(response.text)
    else:
        print("\n⚠️ [ATTENTION] Réponse reçue, mais le champ .text est vide.")

except Exception as e:
    print(f"\n💥 [CRASH LOCAL] L'appel a échoué : {str(e)}")
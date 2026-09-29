"""
Point d'entrée principal de l'application ImmoPredict AI.
Lance l'interface utilisateur conversationnelle Gradio.
"""
import os
import sys
from pathlib import Path

# Force UTF-8 encoding for standard output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Chargement prioritaire du fichier .env à la racine
PROJECT_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

from src.interface.ui import build_app

if __name__ == "__main__":
    print("Démarrage de l'application ImmoPredict AI...")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

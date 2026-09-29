import sys
from pathlib import Path

# Activer l'import de la racine
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.agent import RealEstateAgent

def test():
    agent = RealEstateAgent()
    print("Agent initialisé :", agent.client is not None)
    print("Modèle configuré :", agent.model_name)
    print("Base URL :", agent.base_url)

    res = agent.process_query("Bonjour, que peux-tu faire ?")
    print("Résultat test simple :", res)

if __name__ == "__main__":
    test()

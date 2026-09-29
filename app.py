"""
Point d'entrée Hugging Face Spaces pour ImmoPredict AI.
"""
try:
    import spaces

    @spaces.GPU
    def _dummy_gpu():
        pass
except (ImportError, Exception):
    pass

from src.interface.ui import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

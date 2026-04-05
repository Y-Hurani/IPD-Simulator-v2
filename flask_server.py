"""
flask_server.py
───────────────
Singleton Flask server shared across all Dash apps.
"""

from flask import Flask

_server = Flask(__name__)


def get_server() -> Flask:
    return _server


if __name__ == "__main__":
    _server.run(debug=True)
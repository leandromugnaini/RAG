import os

from flask import Flask, send_from_directory

from app.core.config import settings  # load .env / secrets
from app.routers.documents import documents_bp
from app.routers.question import question_bp


def create_app() -> Flask:
    app = Flask(__name__)
    # global config
    app.config["UPLOAD_DIR"] = settings.UPLOAD_DIR  # e.g. "data/uploads"

    # register blueprints
    app.register_blueprint(documents_bp)
    app.register_blueprint(question_bp)

    # @app.route("/", methods=["GET"])
    # def index():
    #     return jsonify({
    #         "endpoints": ["/documents", "/question"]
    #     })

    @app.route("/")
    def index_html():
        return send_from_directory("static", "index.html")

    # prevent 404 on favicon.ico request
    @app.route("/favicon.ico")
    def favicon():
        return "", 204

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

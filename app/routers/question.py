from flask import Blueprint, jsonify, request

from app.schemas.question import QuestionRequest, QuestionResponse
from app.services import retriever

question_bp = Blueprint("question", __name__, url_prefix="/question")


@question_bp.route("", methods=["POST"])
def ask_question():
    try:
        payload = QuestionRequest(**(request.get_json(force=True) or {}))
    except Exception as exc:
        return jsonify({"detail": str(exc)}), 400

    result = retriever.answer_question(payload.question)
    body = QuestionResponse(**result).model_dump()

    return jsonify(body), 200

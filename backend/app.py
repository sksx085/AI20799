from flask import Flask, request, jsonify
import logging
from services.agent_service import get_rumor_status_and_refutation
from services.statistics import rumor_detection_lock, rumor_detection_count, rumor_detection_history

app = Flask(__name__)

MAX_TEXT_LENGTH = 1000
logging.basicConfig(level=logging.INFO)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON"}), 400

        text = data.get('text', '').strip()
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"success": False, "error": f"Text length exceeds limit of {MAX_TEXT_LENGTH} characters"}), 400

        info = get_rumor_status_and_refutation(text)
        return jsonify({"success": True, "data": info})

    except Exception as e:
        app.logger.error(f"/analyze error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route('/stats', methods=['GET'])
def stats():
    try:
        with rumor_detection_lock:
            return jsonify({
                "total_detection_count": rumor_detection_count,
                "history": rumor_detection_history
            })
    except Exception as e:
        app.logger.error(f"/stats error: {str(e)}", exc_info=True)
        # 返回默认空数据，确保 JSON 格式合法
        return jsonify({
            "total_detection_count": 0,
            "history": []
        })

@app.errorhandler(500)
def handle_500(e):
    app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)

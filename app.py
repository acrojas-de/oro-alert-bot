from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

@app.route("/telegram", methods=["POST"])
def telegram():
    try:
        data = request.get_json(force=True) or {}
        message = data.get("message", "Radar alerta")

        if not BOT_TOKEN or not CHAT_ID:
            return jsonify({"ok": False, "error": "Faltan BOT_TOKEN o CHAT_ID"}), 500

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": message
        }, timeout=15)

        return jsonify({
            "ok": r.ok,
            "telegram_status": r.status_code,
            "telegram_response": r.json()
        }), (200 if r.ok else 500)

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

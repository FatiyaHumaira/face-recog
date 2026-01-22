#!/usr/bin/env python3
"""
Simple mock webhook server for testing webhook integration
Receives and logs webhook payloads from face recognition API
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime
import json

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store received webhooks
received_webhooks = []


@app.route('/api/detection-webhook', methods=['POST'])
def receive_detection():
    """Receive detection webhook from face recognition API"""
    
    try:
        data = request.get_json()
        
        if not data:
            logger.error("Empty webhook payload received")
            return jsonify({"error": "Empty payload"}), 400
        
        # Log the webhook
        webhook_entry = {
            "timestamp": datetime.now().isoformat(),
            "payload": data
        }
        received_webhooks.append(webhook_entry)
        
        # Extract information
        door_id = data.get('water_channel_door_id', 'Unknown')
        detected_persons = data.get('detected_persons', [])
        
        # Pretty print the webhook
        print("\n" + "="*70)
        print(f"📬 WEBHOOK RECEIVED from Door {door_id}")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total persons detected: {len(detected_persons)}")
        
        if detected_persons:
            print(f"\nDetected Persons:")
            for i, person in enumerate(detected_persons, 1):
                is_human = "Human" if person.get('is_human', True) else "Not Human"
                is_known = "KNOWN" if person.get('is_known_person', False) else "UNKNOWN"
                
                print(f"  [{i}] {is_human} | {is_known}")
        else:
            print("  (No persons detected)")
        
        print("="*70 + "\n")
        
        # Log to file
        logger.info(f"Door {door_id}: {len(detected_persons)} persons detected")
        
        # Return success
        return jsonify({"status": "ok", "message": "Webhook received"}), 200
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route('/stats', methods=['GET'])
def stats():
    """Get webhook statistics"""
    total_webhooks = len(received_webhooks)
    total_persons = sum(
        len(w['payload'].get('detected_persons', []))
        for w in received_webhooks
    )
    
    return jsonify({
        "total_webhooks": total_webhooks,
        "total_persons_detected": total_persons,
        "average_persons_per_webhook": (
            total_persons / total_webhooks if total_webhooks > 0 else 0
        )
    })


@app.route('/history', methods=['GET'])
def history():
    """Get last N webhook entries"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        "total_received": len(received_webhooks),
        "last_entries": received_webhooks[-limit:]
    })


@app.route('/clear', methods=['POST'])
def clear():
    """Clear webhook history"""
    global received_webhooks
    count = len(received_webhooks)
    received_webhooks = []
    return jsonify({"message": f"Cleared {count} webhook entries"}), 200


def main():
    print("\n" + "="*70)
    print("MOCK WEBHOOK SERVER FOR FACE RECOGNITION")
    print("="*70)
    print("\nEndpoints:")
    print("  POST   /api/detection-webhook  - Receive detection webhook")
    print("  GET    /health                 - Health check")
    print("  GET    /stats                  - Webhook statistics")
    print("  GET    /history?limit=10       - View webhook history")
    print("  POST   /clear                  - Clear history")
    print("\nConfiguration:")
    print("  Host: 127.0.0.1")
    print("  Port: 9000")
    print("\nTo use with API:")
    print("  export WEBHOOK_ENABLED=true")
    print("  export WEBHOOK_URL=http://localhost:9000/api/detection-webhook")
    print("  python run_api.py")
    print("\nThen test:")
    print("  curl -X POST http://localhost:8000/facerecognizer \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"image_url\":\"...\", \"water_channel_door_id\":\"DOOR_001\"}'")
    print("\n" + "="*70 + "\n")
    
    # Start server
    app.run(
        host='127.0.0.1',
        port=9000,
        debug=True,
        use_reloader=False
    )


if __name__ == '__main__':
    main()

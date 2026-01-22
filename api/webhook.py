import requests
import logging
from typing import Union, List
from api.models import DetectedPerson, DetectionWebhookReq
from config import Config

logger = logging.getLogger(__name__)


def transform_to_webhook_format(
    recognized_ids: Union[List[str], str],
    water_channel_door_id: str,
    door_id_mapping: dict = None
) -> DetectionWebhookReq:
    """
    Transform /facerecognizer response ke webhook format
    
    Args:
        recognized_ids: Array atau string dari recognized IDs
        water_channel_door_id: Door ID string (e.g., "DOOR_001")
        door_id_mapping: Optional dict untuk map DOOR_001 -> 1
    
    Returns:
        DetectionWebhookReq object
    """
    
    # Convert door_id_str ke numeric
    try:
        # Extract numeric part dari door_id string
        # e.g., "DOOR_001" -> 1, "278" -> 278
        door_id_numeric = int(
            ''.join(filter(str.isdigit, water_channel_door_id)) or "0"
        )
    except:
        door_id_numeric = 0
    
    # Transform recognized_ids ke DetectedPerson array
    detected_persons = []
    
    # Handle different response formats
    if isinstance(recognized_ids, str):
        # Single string: "EMP_001" atau "unknown" atau "0"
        if recognized_ids == "0":
            # No faces detected - empty array
            pass  # Empty detected_persons
        elif recognized_ids.lower() == "unknown":
            # Unknown person detected
            person = DetectedPerson(
                is_human=True,
                is_known_person=False
            )
            detected_persons.append(person)
        else:
            # Single recognized person
            person = DetectedPerson(
                is_human=True,
                is_known_person=True
            )
            detected_persons.append(person)
    
    elif isinstance(recognized_ids, list):
        # Multiple persons
        for person_id in recognized_ids:
            if person_id.lower() == "unknown":
                # Unknown person
                person = DetectedPerson(
                    is_human=True,
                    is_known_person=False
                )
            else:
                # Known person
                person = DetectedPerson(
                    is_human=True,
                    is_known_person=True
                )
            detected_persons.append(person)
    
    return DetectionWebhookReq(
        water_channel_door_id=door_id_numeric,
        detected_persons=detected_persons
    )


async def send_webhook(webhook_data: DetectionWebhookReq) -> bool:
    """
    Send detection results ke downstream webhook endpoint
    
    Args:
        webhook_data: DetectionWebhookReq object
    
    Returns:
        bool: True jika success, False jika failed
    """
    
    if not Config.WEBHOOK_ENABLED:
        logger.debug("Webhook disabled")
        return True
    
    try:
        logger.info(f"Sending webhook to: {Config.WEBHOOK_URL}")
        
        response = requests.post(
            Config.WEBHOOK_URL,
            json=webhook_data.dict(),
            timeout=Config.WEBHOOK_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info(f"✓ Webhook sent successfully: {response.status_code}")
            return True
        else:
            logger.warning(f"✗ Webhook failed: {response.status_code} - {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        logger.error(f"✗ Webhook timeout ({Config.WEBHOOK_TIMEOUT}s)")
        return False
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"✗ Webhook connection error: {e}")
        return False
    
    except Exception as e:
        logger.error(f"✗ Webhook error: {e}")
        return False

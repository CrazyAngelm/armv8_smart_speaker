import pytest
from unittest.mock import MagicMock
import json
import threading
from datetime import datetime, timedelta
import time

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    event_checker,
    events,
    SetTimerEvent,
    SetNotificationEvent,
    stop_event_checker,
    set_time_handler,
    get_time_handler,
    set_notification_handler,
)


@pytest.fixture
def mock_client():

    client = MagicMock()
    return client


@pytest.fixture
def nlu_payload_set_timer():

    return {
        "intent": {
            "intentName": "SetTimer"
        },
        "slots": [
            {"slotName": "hour", "value": {"value": 0}},
            {"slotName": "minute", "value": {"value": 0}},
            {"slotName": "second", "value": {"value": 2}}
        ]
    }


@pytest.fixture
def nlu_payload_set_notification():

    return {
        "intent": {
            "intentName": "SetNotification"
        },
        "slots": [
            {"slotName": "hour", "value": {"value": 0}},
            {"slotName": "minute", "value": {"value": 0}},
            {"slotName": "second", "value": {"value": 2}}
        ],
        "rawInput": "Напомни мне о том, что нужно позвонить"
    }


@pytest.fixture
def event_checker_thread(mock_client):

    stop_event_checker.clear()
    events.clear()
    thread = threading.Thread(target=event_checker, args=(mock_client,), daemon=True)
    thread.start()
    yield thread
    stop_event_checker.set()
    thread.join(timeout=1)


def test_set_time_handler(mock_client, nlu_payload_set_timer):

    events.clear() 
    set_time_handler(nlu_payload_set_timer, mock_client)

    assert len(events) == 1
    assert isinstance(events[0], SetTimerEvent)

    expected_time = datetime.now() + timedelta(seconds=2)
    assert abs((events[0].timestamp - expected_time).total_seconds()) < 1

    mock_client.publish.assert_called_once_with("hermes/tts/say", json.dumps({"text": "Поставил таймер"}))


def test_get_time_handler(mock_client):

    get_time_handler(mock_client)

    mock_client.publish.assert_called_once()
    args, kwargs = mock_client.publish.call_args
    assert args[0] == "hermes/tts/say"
    assert "Текущее время" in json.loads(args[1])["text"]


def test_set_notification_handler(mock_client, nlu_payload_set_notification):

    events.clear() 
    set_notification_handler(nlu_payload_set_notification, mock_client)

    assert len(events) == 1
    assert isinstance(events[0], SetNotificationEvent)

    assert events[0].notification_text == ", что нужно позвонить"

    expected_time = datetime.now() + timedelta(seconds=2)
    assert abs((events[0].timestamp - expected_time).total_seconds()) < 1

    mock_client.publish.assert_called_once_with("hermes/tts/say", json.dumps({"text": "Поставил напоминание"}))


def test_event_checker(mock_client, event_checker_thread):

    events.clear()
    event_time = datetime.now() + timedelta(seconds=2)
    event = SetTimerEvent(timestamp=event_time)
    events.append(event)

    time.sleep(3)

    assert len(events) == 0
    mock_client.publish.assert_called_with("hermes/tts/say", json.dumps({"text": event.finish_event()}))

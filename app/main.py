import asyncio
import http
import json
import logging
import os
import sys
import time
import websockets
from websockets import ClientConnection

from bandwidth import Configuration, ApiClient, CallsApi
from bandwidth.models import InitiateCallback, DisconnectCallback
from bandwidth.models.bxml import PhoneNumber, Transfer, Bxml
from rich import inspect
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from fastapi import FastAPI, Response, WebSocket
import uvicorn

from models import BandwidthStreamEvent, StreamEventType, StreamMedia
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
# In production (Render), environment variables are injected directly
load_dotenv(dotenv_path="../.env")

console = Console()
try:
    BW_ACCOUNT = os.environ["BW_ACCOUNT_ID"]
    BW_USERNAME = os.environ["BW_USERNAME"]
    BW_PASSWORD = os.environ["BW_PASSWORD"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    TRANSFER_TO = os.environ["TRANSFER_TO"]
    BASE_URL = os.environ["BASE_URL"]
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()  # Default to INFO if not set
    LOCAL_PORT = int(os.environ.get("PORT", os.environ.get("LOCAL_PORT", 3000)))
except KeyError as e:
    msg = Text(" Missing environment variables! ", style="bold white on red")
    details = f"Required key not set: [yellow]{e.args[0]}[/yellow]\n\n"
    details += "Make sure the following variables are defined:\n"
    details += "[cyan]BW_ACCOUNT, BW_USERNAME, BW_PASSWORD, OPENAI_API_KEY, OPENAI_SIGNING_SECRET, TRANSFER_TO, BASE_URL, LOG_LEVEL, PORT[/cyan]"
    console.print(Panel(details, title=msg, expand=False, border_style="red"))
    sys.exit(1)

# Configure Logger
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)s %(asctime)s: %(message)s",
    datefmt="[%X]",
)
for name in ["websockets", "asyncio", "urllib3", "uvicorn", "fastapi"]:
    # Set specific loggers to INFO to reduce noise
    logging.getLogger(name).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Bandwidth Client
bandwidth_config = Configuration(
    username=BW_USERNAME,
    password=BW_PASSWORD
)
bandwidth_client = ApiClient(bandwidth_config)
bandwidth_voice_api_instance = CallsApi(bandwidth_client)

# OpenAI Agent Settings
AGENT_TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))
AGENT_VOICE = "alloy"
AGENT_GREETING = "Hi welcome to bandwidth, I'm an ai agent who help answer your questions"
CALL_KEEPALIVE_SECONDS = int(os.environ.get("CALL_KEEPALIVE_SECONDS", 600))
BANDWIDTH_EVENT_FIELD = os.environ.get("BANDWIDTH_EVENT_FIELD", "eventType")
BANDWIDTH_AUDIO_FIELD = os.environ.get("BANDWIDTH_AUDIO_FIELD", "media")
BANDWIDTH_AUDIO_CONTENT_TYPE = os.environ.get("BANDWIDTH_AUDIO_CONTENT_TYPE", "audio/pcmu")
ECHO_AUDIO = os.environ.get("ECHO_AUDIO", "false").strip().lower() == "true"

# Load prompt from file (handle both local and production paths)
try:
    with open("sample-prompt.md", "r", encoding="utf-8") as file:
        AGENT_PROMPT = file.read()
except FileNotFoundError:
    # Fallback prompt if file not found
    AGENT_PROMPT = "You are a helpful AI assistant. Be friendly and professional."
    print("WARNING: sample-prompt.md not found, using default prompt")


# Initialize FastAPI app
app = FastAPI(title="Bandwidth + OpenAI Realtime Voice Agent", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 80)
    logger.info("Bandwidth + OpenAI Realtime Voice Agent Started")
    logger.info(f"Base URL: {BASE_URL}")
    logger.info(f"Port: {LOCAL_PORT}")
    logger.info(f"Environment: {'Production' if os.environ.get('PORT') else 'Development'}")
    logger.info("=" * 80)


def log_inspect(obj, label=None):
    """
    Log the inspection of an object if debug logging is enabled.
    :param obj: The object to inspect
    :param label: An optional label for the inspection
    :return: None
    """
    if logger.isEnabledFor(logging.DEBUG):
        inspect(
            obj,
            title=label or repr(obj)
        )


def build_bandwidth_event(
    event_type: str,
    call_id: str | None = None,
    stream_id: str | None = None,
    media_payload: str | None = None,
    content_type: str | None = None,
) -> dict:
    event = {BANDWIDTH_EVENT_FIELD: event_type}
    if call_id:
        event["callId"] = call_id
    if stream_id:
        event["streamId"] = stream_id
    if media_payload is not None:
        media_obj = {
            "payload": media_payload,
            "contentType": content_type or BANDWIDTH_AUDIO_CONTENT_TYPE,
        }
        event[BANDWIDTH_AUDIO_FIELD] = media_obj
    return event


async def send_bandwidth_event(
    bandwidth_websocket: WebSocket,
    event: BandwidthStreamEvent | dict,
    label: str,
) -> None:
    try:
        # Check if WebSocket is still connected before sending
        if bandwidth_websocket.client_state.name == "DISCONNECTED":
            logger.debug(f"Skipping Bandwidth event ({label}) - WebSocket already disconnected")
            return
            
        if isinstance(event, BandwidthStreamEvent):
            event_json = event.model_dump_json(by_alias=True, exclude_none=True)
        else:
            event_json = json.dumps(event)
        await bandwidth_websocket.send_text(event_json)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Sent Bandwidth event: %s - %s", label, event_json[:200])
        else:
            logger.info("Sent Bandwidth event: %s", label)
    except RuntimeError as e:
        # FastAPI WebSocket might raise RuntimeError if already closed
        error_msg = str(e).lower()
        if "close" in error_msg or "disconnected" in error_msg:
            logger.debug(f"Failed to send Bandwidth event ({label}) - WebSocket already closed")
        else:
            logger.error("Failed to send Bandwidth event (%s): %s", label, e, exc_info=True)
    except Exception as e:
        # Suppress errors for already-closed connections
        error_msg = str(e).lower()
        if "close" in error_msg and ("already" in error_msg or "sent" in error_msg or "disconnected" in error_msg):
            logger.debug(f"Failed to send Bandwidth event ({label}) - WebSocket already closed")
        else:
            logger.error("Failed to send Bandwidth event (%s): %s", label, e, exc_info=True)


async def initialize_openai_session(websocket: ClientConnection):
    """
    Initialize the OpenAI WebSocket session with the required configuration and initial message.
    :param websocket:
    :return: None
    """
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": AGENT_PROMPT,
            "voice": AGENT_VOICE,
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            },
            "tools": [
                {
                    "type": "function",
                    "name": "transfer_call",
                    "description": "Call this function when the user asks to be transferred.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ],
            "tool_choice": "auto",
            "temperature": AGENT_TEMPERATURE
        }
    }
    await websocket.send(json.dumps(session_update))
    logger.info(f"Sent Session Update event to OpenAI")

    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Greet the user with '{AGENT_GREETING}'"
                }
            ]
        }
    }
    await websocket.send(json.dumps(initial_conversation_item))
    await websocket.send(json.dumps({
        "type": "response.create",
        "response": {
            "modalities": ["audio", "text"]
        }
    }))
    logger.info(f"Sent initial greeting request to OpenAI")
    

async def receive_from_bandwidth_ws(
    bandwidth_websocket: WebSocket,
    openai_websocket: ClientConnection,
    stream_context: dict,
):
    """
    Receive messages from Bandwidth WebSocket and forward audio to OpenAI WebSocket.
    :param bandwidth_websocket:
    :param openai_websocket:
    :return: None
    """
    media_count = 0
    stream_started_time = None
    last_event_time = None
    try:
        logger.info("Starting to listen for Bandwidth messages...")
        logger.info(f"Bandwidth WebSocket state: {bandwidth_websocket.client_state.name}")
        logger.info(f"OpenAI WebSocket state: {openai_websocket.state if hasattr(openai_websocket, 'state') else 'unknown'}")
        
        async for message in bandwidth_websocket.iter_json():
            last_event_time = time.time()
            logger.info(f"Raw Bandwidth message received: {json.dumps(message)[:200]}...")
            
            try:
                event = BandwidthStreamEvent.model_validate(message)
                stream_id = event.metadata.stream_id if event.metadata else None
                call_id = event.call_id or (event.metadata.call_id if event.metadata else None)
                logger.info(f"Parsed Bandwidth event: {event.event_type} (call_id={call_id}, stream_id={stream_id})")
            except Exception as parse_error:
                logger.error(f"Failed to parse Bandwidth event: {parse_error}")
                logger.error(f"Raw message: {json.dumps(message)}")
                continue
                
            match event.event_type:
                case StreamEventType.STREAM_STARTED:
                    stream_started_time = time.time()
                    logger.info(f"✓ Stream started for call ID: {event.metadata.call_id if event.metadata else 'unknown'}")
                    if event.metadata:
                        logger.info(
                            "Stream metadata: stream_id=%s stream_name=%s",
                            event.metadata.stream_id,
                            event.metadata.stream_name,
                        )
                        stream_context["stream_id"] = event.metadata.stream_id
                        stream_context["stream_name"] = event.metadata.stream_name
                        if event.metadata.tracks:
                            logger.info(f"Number of tracks: {len(event.metadata.tracks)}")
                            for track in event.metadata.tracks:
                                media_format = track.media_format
                                logger.info(
                                    "Track: name=%s encoding=%s sample_rate=%s",
                                    track.name,
                                    media_format.encoding if media_format else None,
                                    media_format.sample_rate if media_format else None,
                                )
                        else:
                            logger.warning("No tracks found in stream metadata")
                    else:
                        logger.warning("No metadata in STREAM_STARTED event")
                case StreamEventType.MEDIA:
                    media_count += 1
                    payload = event.payload
                    if not payload and event.media:
                        payload = event.media.payload
                    content_type = event.media.content_type if event.media else None
                    if not payload:
                        logger.warning("Media event missing payload")
                        continue
                    payload_len = len(payload)
                    approx_bytes = (payload_len * 3) // 4
                    approx_ms = int(approx_bytes / 8) if approx_bytes else 0  # 8kHz u-law
                    if media_count <= 3 or media_count % 50 == 0:
                        logger.info(
                            "Media packet %s: b64_len=%s bytes≈%s audio≈%sms content_type=%s",
                            media_count,
                            payload_len,
                            approx_bytes,
                            approx_ms,
                            content_type,
                        )
                    audio_append = {
                        "type": "input_audio_buffer.append",
                        "audio": payload
                    }
                    await openai_websocket.send(json.dumps(audio_append))
                    if ECHO_AUDIO:
                        if not stream_context.get("echo_logged"):
                            logger.info("Echo mode enabled: sending inbound audio back to caller")
                            stream_context["echo_logged"] = True
                        echo_event = BandwidthStreamEvent(
                            event_type=StreamEventType.PLAY_AUDIO,
                            call_id=stream_context.get("call_id"),
                            media=StreamMedia(
                                payload=payload,
                                content_type=content_type or BANDWIDTH_AUDIO_CONTENT_TYPE
                            )
                        )
                        await send_bandwidth_event(bandwidth_websocket, echo_event, "echo playAudio")
                case StreamEventType.STREAM_STOPPED:
                    stream_duration = None
                    if stream_started_time:
                        stream_duration = time.time() - stream_started_time
                    
                    # Extract stream_id from metadata if available
                    stream_id_from_event = event.metadata.stream_id if event.metadata else None
                    if stream_id_from_event and not stream_context.get("stream_id"):
                        stream_context["stream_id"] = stream_id_from_event
                    
                    if media_count == 0:
                        logger.warning("=" * 80)
                        logger.warning("Stream stopped without receiving any media packets")
                        logger.warning(f"Call ID: {stream_context.get('call_id', 'unknown')}")
                        logger.warning(f"Stream ID: {stream_context.get('stream_id', 'unknown')}")
                        logger.warning(f"Stream duration: {stream_duration:.2f}s" if stream_duration else "Stream duration: unknown")
                        logger.warning(f"Time since last event: {time.time() - last_event_time:.2f}s" if last_event_time else "N/A")
                        logger.warning("NOTE: This is normal if the caller didn't speak or the call ended before audio was captured")
                        logger.warning("=" * 80)
                    else:
                        logger.info(f"Stream stopped after {media_count} media packets (duration: {stream_duration:.2f}s)" if stream_duration else f"Stream stopped after {media_count} media packets")
                    break
                case _:
                    logger.warning(f"Unhandled event type: {event.event_type}")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error("=" * 80)
        logger.error("Bandwidth WebSocket ConnectionClosedError")
        logger.error(f"Error: {e}")
        logger.error(f"Code: {e.code if hasattr(e, 'code') else 'unknown'}")
        logger.error(f"Reason: {e.reason if hasattr(e, 'reason') else 'unknown'}")
        logger.error(f"Media packets received: {media_count}")
        logger.error(f"Stream started: {stream_started_time is not None}")
        logger.error(f"Bandwidth WebSocket state: {bandwidth_websocket.client_state.name}")
        logger.error("=" * 80)
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Exception in receive_from_bandwidth_ws")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Media packets received: {media_count}")
        logger.error(f"Stream started: {stream_started_time is not None}")
        logger.error("=" * 80)
        logger.error(f"Full traceback:", exc_info=True)
    finally:
        # Don't close WebSockets here - let the main handler do it to avoid race conditions
        logger.info(f"Bandwidth receive task completed (state: {bandwidth_websocket.client_state.name})")


async def receive_from_openai_ws(
    openai_websocket: ClientConnection,
    bandwidth_websocket: WebSocket,
    call_id: str,
    stream_context: dict,
):
    """
    Receive messages from OpenAI WebSocket and forward audio to Bandwidth WebSocket.
    :param openai_websocket:
    :param bandwidth_websocket:
    :param call_id:
    :return: None
    """
    last_assistant_item = None
    audio_delta_count = 0
    try:
        logger.info("Starting to listen for OpenAI messages...")
        logger.info(f"OpenAI WebSocket state: {openai_websocket.state if hasattr(openai_websocket, 'state') else 'unknown'}")
        async for message in openai_websocket:
            openai_message = json.loads(message)
            message_type = openai_message.get('type')
            logger.debug(f"OpenAI message: {message_type}")
            transcript = openai_message.get("transcript")
            if transcript:
                if "input_audio" in (message_type or ""):
                    logger.info(f"User transcript: {transcript}")
                elif "output_audio" in (message_type or ""):
                    logger.info(f"Assistant transcript: {transcript}")
                else:
                    logger.info(f"Transcript ({message_type}): {transcript}")
            
            # Handle any audio delta event types (OpenAI has used multiple names)
            if (
                message_type
                and message_type.endswith(".delta")
                and "audio" in message_type
                and "transcript" not in message_type
                and openai_message.get("delta")
            ):
                audio_delta_count += 1
                audio_payload = openai_message["delta"]
                if audio_delta_count <= 5 or audio_delta_count % 50 == 0:
                    logger.info(
                        "OpenAI audio delta %s: b64_len=%s",
                        audio_delta_count,
                        len(audio_payload),
                    )
                # Create playAudio event using BandwidthStreamEvent model for proper serialization
                # Include stream_id from context if available (some Bandwidth configurations may require it)
                stream_id = stream_context.get("stream_id")
                play_audio_event = BandwidthStreamEvent(
                    event_type=StreamEventType.PLAY_AUDIO,
                    call_id=call_id,
                    media=StreamMedia(
                        payload=audio_payload,
                        content_type=BANDWIDTH_AUDIO_CONTENT_TYPE
                    )
                )
                # Build the event dict manually to include streamId if we have it
                event_dict = play_audio_event.model_dump(by_alias=True, exclude_none=True)
                if stream_id:
                    event_dict["streamId"] = stream_id
                await send_bandwidth_event(bandwidth_websocket, event_dict, f"playAudio (delta {audio_delta_count})")
                continue

            match message_type:
                case 'session.created' | 'session.updated':
                    logger.info(f"OpenAI session event: {message_type}")
                case 'response.created':
                    logger.debug(f"Response created: {message_type}")
                case 'response.done':
                    logger.info(f"Response done - ready for next turn")
                case 'conversation.item.done':
                    if openai_message.get('item').get('type') == 'function_call':
                        function_name = openai_message.get('item').get('name')
                        handle_tool_call(function_name, call_id)
                case 'input_audio_buffer.speech_started':
                    logger.info("User speech detected")
                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": 0
                    }
                    await openai_websocket.send(json.dumps(truncate_event))
                    clear_event = BandwidthStreamEvent(
                        event_type=StreamEventType.CLEAR,
                        call_id=call_id
                    )
                    await send_bandwidth_event(bandwidth_websocket, clear_event, "clear")
                    last_assistant_item = None
                case 'error':
                    logger.error(f"OpenAI Error: {openai_message.get('error').get('message')}")
                case _:
                    logger.debug(f"Unhandled OpenAI message type: {openai_message.get('type')}")
                    pass
            if openai_message.get('item'):
                try:
                    last_assistant_item = openai_message.get('item').get('id')
                except KeyError:
                    pass
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error("=" * 80)
        logger.error("OpenAI WebSocket ConnectionClosedError")
        logger.error(f"Error: {e}")
        logger.error(f"Code: {e.code if hasattr(e, 'code') else 'unknown'}")
        logger.error(f"Reason: {e.reason if hasattr(e, 'reason') else 'unknown'}")
        logger.error(f"Audio deltas sent: {audio_delta_count}")
        logger.error(f"Call ID: {call_id}")
        logger.error(f"OpenAI WebSocket state: {openai_websocket.state if hasattr(openai_websocket, 'state') else 'unknown'}")
        logger.error("=" * 80)
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Exception in receive_from_openai_ws")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Audio deltas sent: {audio_delta_count}")
        logger.error(f"Call ID: {call_id}")
        logger.error("=" * 80)
        logger.error("Full traceback:", exc_info=True)
    finally:
        # Don't close WebSockets here - let the main handler do it to avoid race conditions
        logger.info(f"OpenAI receive task completed (state: {openai_websocket.state if hasattr(openai_websocket, 'state') else 'unknown'})")


def handle_tool_call(function_name: str, call_id: str = None):
    """
    Handle tool calls from OpenAI by executing the corresponding Bandwidth API actions.
    :param function_name:
    :param call_id:
    :return: None
    """
    match function_name:
        case 'transfer_call':
            logger.info("Request to transfer_call received")
            transfer_number = PhoneNumber(TRANSFER_TO)
            transfer_bxml = Transfer([transfer_number])
            update_call_bxml = Bxml([transfer_bxml])
            try:
                bandwidth_voice_api_instance.update_call_bxml(BW_ACCOUNT, call_id, update_call_bxml.to_bxml())
            except Exception as e:
                logger.error(f"Error transferring call: {e}")
                logger.warning(f"Unhandled function call: {function_name}")
    return


@app.get("/health", status_code=http.HTTPStatus.NO_CONTENT)
def health():
    """
    Health check endpoint
    :return: None
    """
    return


@app.get("/")
def root():
    """
    Root endpoint for platform health checks.
    Returns 200 so hosting providers (e.g., Render) don't treat the app as unhealthy.
    """
    return {"status": "ok"}


@app.get("/status")
def status():
    """
    Status endpoint showing configuration details
    :return: JSON with status information
    """
    return {
        "status": "running",
        "base_url": BASE_URL,
        "local_port": LOCAL_PORT,
        "log_level": LOG_LEVEL,
        "openai_configured": bool(OPENAI_API_KEY),
        "bandwidth_configured": bool(BW_ACCOUNT and BW_USERNAME and BW_PASSWORD),
        "transfer_to": TRANSFER_TO,
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "voice_initiate": "/webhooks/bandwidth/voice/initiate",
            "voice_status": "/webhooks/bandwidth/voice/status",
            "websocket": "/ws"
        }
    }


@app.post("/webhooks/bandwidth/voice/initiate", status_code=http.HTTPStatus.OK)
def handle_initiate_event(callback: InitiateCallback) -> Response:
    """
    Handle the initiate event from Bandwidth

    :param callback: The initiate callback data
    :return: An empty response with status code 200
    """
    # log_inspect(callback, label="Initiate Callback")
    call_id = callback.call_id
    logger.info(f"Received initiate event for call ID: {call_id}")

    websocket_url = f"wss://{BASE_URL.replace('https://', '').replace('http://', '')}/ws"
    destination = f"{websocket_url}?call_id={call_id}"
    # Use SpeakSentence before StartStream to give time for OpenAI connection
    # and to let the caller know the AI is connecting
    # Important: use Bandwidth's StartStream 'tracks' attribute (mediaTrack enum) instead of a Twilio-style "mode" flag.
    # Valid values are: 'inbound', 'outbound', or 'both'.
    # 'both' gives you:
    #   - inbound: caller -> this WebSocket (so we see MEDIA events and your speech)
    #   - outbound: this WebSocket -> caller (so playAudio events are actually heard)
    bxml_content = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "\n<Bxml>"
        "<SpeakSentence voice=\"julie\">Please wait while I connect you to the AI assistant.</SpeakSentence>"
        f"<StartStream destination=\"{destination}\" name=\"{call_id}\" tracks=\"both\" />"
        f"<Pause duration=\"600\" />"
        "</Bxml>"
    )
    logger.info(f"Sending BXML for call {call_id}: {bxml_content}")
    return Response(status_code=http.HTTPStatus.OK, content=bxml_content, media_type="application/xml")


async def connect_and_init_openai(call_id: str):
    """Connect to OpenAI and initialize the session"""
    try:
        openai_websocket = await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            },
            open_timeout=5,
            ping_interval=None
        )
        logger.info(f"✓ Connected to OpenAI WebSocket for call ID: {call_id}")
        await initialize_openai_session(openai_websocket)
        logger.info(f"✓ OpenAI session initialized for call ID: {call_id}")
        return openai_websocket
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI: {e}")
        raise


@app.websocket("/ws")
async def handle_inbound_websocket(bandwidth_websocket: WebSocket, call_id: str = None):
    """
    Handle inbound WebSocket connections from Bandwidth and bridge to OpenAI WebSocket.
    :param bandwidth_websocket:
    :param call_id:
    :return: None
    """
    connection_start_time = time.time()
    openai_websocket = None
    stream_context = {"call_id": call_id, "stream_id": None, "stream_name": None, "echo_logged": False}
    
    try:
        logger.info("=" * 80)
        logger.info(f"Incoming WebSocket connection request for call ID: {call_id}")
        logger.info(f"WebSocket URL query params: {bandwidth_websocket.query_params}")
        logger.info(f"WebSocket headers: {dict(bandwidth_websocket.headers)}")
        
        await bandwidth_websocket.accept()
        logger.info(f"✓ Bandwidth WebSocket connection accepted for call ID: {call_id}")
        logger.info(f"WebSocket state after accept: {bandwidth_websocket.client_state.name}")

        if not call_id:
            logger.error("No call_id provided in WebSocket connection")
            await bandwidth_websocket.close(code=1008, reason="Missing call_id parameter")
            return

        # Start OpenAI connection immediately but don't wait for it
        logger.info(f"Starting OpenAI connection task for call ID: {call_id}")
        openai_task = asyncio.create_task(connect_and_init_openai(call_id))
        
        # Start consuming Bandwidth messages immediately to prevent timeout
        try:
            logger.info(f"Waiting for first message from Bandwidth (timeout: 5s)...")
            # Wait for first message from Bandwidth (should be stream_started)
            first_message = await asyncio.wait_for(bandwidth_websocket.receive_json(), timeout=5.0)
            logger.info(f"First message received after {time.time() - connection_start_time:.2f}s")
            logger.info(f"First message content: {json.dumps(first_message)[:500]}")
            
            try:
                event = BandwidthStreamEvent.model_validate(first_message)
                logger.info(f"✓ First Bandwidth event parsed: {event.event_type}")
            except Exception as parse_error:
                logger.error(f"Failed to parse first Bandwidth event: {parse_error}")
                logger.error(f"Raw first message: {json.dumps(first_message)}")
                raise
            
            # Now wait for OpenAI to be ready (should be quick)
            logger.info(f"Waiting for OpenAI connection to complete (timeout: 3s)...")
            try:
                openai_websocket = await asyncio.wait_for(openai_task, timeout=3.0)
                logger.info(f"✓ OpenAI connection ready after {time.time() - connection_start_time:.2f}s")
            except asyncio.TimeoutError:
                logger.error("=" * 80)
                logger.error(f"OpenAI connection timeout for call {call_id}")
                logger.error(f"Time elapsed: {time.time() - connection_start_time:.2f}s")
                logger.error("=" * 80)
                await bandwidth_websocket.close(code=1011, reason="OpenAI connection timeout")
                return
            
            # Now both are connected, start bidirectional streaming
            logger.info("=" * 80)
            logger.info("Starting bidirectional streaming between Bandwidth and OpenAI")
            logger.info(f"Total setup time: {time.time() - connection_start_time:.2f}s")
            logger.info("=" * 80)
            
            await asyncio.gather(
                receive_from_bandwidth_ws(bandwidth_websocket, openai_websocket, stream_context),
                receive_from_openai_ws(openai_websocket, bandwidth_websocket, call_id, stream_context)
            )
                
        except asyncio.TimeoutError:
            logger.error("=" * 80)
            logger.error(f"Timeout waiting for Bandwidth stream start for call {call_id}")
            logger.error(f"Time elapsed: {time.time() - connection_start_time:.2f}s")
            logger.error(f"WebSocket state: {bandwidth_websocket.client_state.name}")
            logger.error("=" * 80)
            await bandwidth_websocket.close(code=1011, reason="Stream start timeout")
            return
            
    except websockets.exceptions.InvalidStatusCode as e:
        logger.error("=" * 80)
        logger.error("OpenAI WebSocket InvalidStatusCode exception")
        logger.error(f"Status code: {e.status_code}")
        logger.error(f"Error: {e}")
        logger.error(f"Call ID: {call_id}")
        logger.error(f"Time elapsed: {time.time() - connection_start_time:.2f}s")
        logger.error("=" * 80)
        try:
            await bandwidth_websocket.close(code=1011, reason="OpenAI connection failed")
        except Exception as close_error:
            logger.warning(f"Error closing Bandwidth WebSocket: {close_error}")
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Exception in handle_inbound_websocket")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Call ID: {call_id}")
        logger.error(f"Time elapsed: {time.time() - connection_start_time:.2f}s")
        logger.error(f"Bandwidth WebSocket state: {bandwidth_websocket.client_state.name}")
        logger.error("=" * 80)
        logger.error("Full traceback:", exc_info=True)
        try:
            await bandwidth_websocket.close(code=1011, reason="Internal server error")
        except Exception as close_error:
            logger.warning(f"Error closing Bandwidth WebSocket: {close_error}")
    finally:
        # Cleanup - only close WebSockets that are still open
        logger.info(f"WebSocket handler cleanup for call {call_id}")
        logger.info(f"Total connection duration: {time.time() - connection_start_time:.2f}s")
        
        # Close OpenAI WebSocket if it exists and is still open
        if openai_websocket:
            try:
                # Check if WebSocket is already closed by checking state value
                if hasattr(openai_websocket, 'state'):
                    # State is an integer: 0=CONNECTING, 1=OPEN, 2=CLOSING, 3=CLOSED
                    state = openai_websocket.state
                    if state >= 2:  # CLOSING or CLOSED
                        logger.debug("OpenAI WebSocket already closed or closing")
                    else:
                        logger.info("Closing OpenAI WebSocket in finally block")
                        await openai_websocket.close()
                else:
                    # If we can't check state, try to close but catch exceptions
                    logger.info("Closing OpenAI WebSocket in finally block")
                    await openai_websocket.close()
            except websockets.exceptions.ConnectionClosed:
                logger.debug("OpenAI WebSocket was already closed")
            except Exception as e:
                # Suppress warnings for already-closed connections
                error_msg = str(e).lower()
                if "close" in error_msg and ("already" in error_msg or "sent" in error_msg):
                    logger.debug(f"OpenAI WebSocket already closed: {e}")
                else:
                    logger.warning(f"Error closing OpenAI WebSocket in finally: {e}")
        
        # Close Bandwidth WebSocket if it's still connected
        try:
            bw_state = bandwidth_websocket.client_state.name
            if bw_state == "DISCONNECTED":
                logger.debug("Bandwidth WebSocket already disconnected")
            else:
                logger.info("Closing Bandwidth WebSocket in finally block")
                await bandwidth_websocket.close()
        except RuntimeError as e:
            # FastAPI WebSocket might raise RuntimeError if already closed
            error_msg = str(e).lower()
            if "close" in error_msg or "disconnected" in error_msg:
                logger.debug(f"Bandwidth WebSocket already closed: {e}")
            else:
                logger.warning(f"Error closing Bandwidth WebSocket in finally: {e}")
        except Exception as e:
            # Suppress warnings for already-closed connections
            error_msg = str(e).lower()
            if "close" in error_msg and ("already" in error_msg or "sent" in error_msg or "disconnected" in error_msg):
                logger.debug(f"Bandwidth WebSocket already closed: {e}")
            else:
                logger.warning(f"Error closing Bandwidth WebSocket in finally: {e}")


@app.post("/webhooks/bandwidth/voice/status", status_code=http.HTTPStatus.NO_CONTENT)
def handle_disconnect_event(callback: DisconnectCallback) -> None:
    """
    Handle call status events from Bandwidth

    :param callback: The disconnect callback data
    :return: None
    """
    # log_inspect(callback, label="Disconnect Callback")
    call_id = callback.call_id
    disconnect_cause = callback.cause
    error_message = callback.error_message
    logger.info("=" * 80)
    logger.info("Bandwidth Disconnect Event Received")
    logger.info(f"Call ID: {call_id}")
    logger.info(f"Disconnect cause: {disconnect_cause}")
    logger.info(f"Error message: {error_message}")
    logger.info(f"Full callback data: {callback}")
    logger.info("=" * 80)
    return


def start_server(port: int) -> None:
    """
    Start the FastAPI server

    :param port: The port to run the server on
    :return: None
    """
    logger.info(f"Starting server on port {port}")
    logger.info(f"Base URL: {BASE_URL}")
    logger.info(f"Log Level: {LOG_LEVEL}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    start_server(LOCAL_PORT)

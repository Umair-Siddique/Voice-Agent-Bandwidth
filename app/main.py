import asyncio
import base64
import http
import json
import logging
import os
import sys
import websockets
from websockets import ClientConnection

from bandwidth import Configuration, ApiClient, CallsApi
from bandwidth.models import InitiateCallback, DisconnectCallback
from bandwidth.models.bxml import PhoneNumber, StartStream, Transfer, Bxml
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
AGENT_GREETING = "Howdy Partner! I'm your AI assistant. How can I help you today?"

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
    await websocket.send(json.dumps({"type": "response.create"}))
    logger.info(f"Sent initial greeting request to OpenAI")
    

async def receive_from_bandwidth_ws(bandwidth_websocket: WebSocket, openai_websocket: ClientConnection):
    """
    Receive messages from Bandwidth WebSocket and forward audio to OpenAI WebSocket.
    :param bandwidth_websocket:
    :param openai_websocket:
    :return: None
    """
    try:
        async for message in bandwidth_websocket.iter_json():
            event = BandwidthStreamEvent.model_validate(message)
            match event.event_type:
                case StreamEventType.STREAM_STARTED:
                    logger.info(f"Stream started for call ID: {event.metadata.call_id}")
                case StreamEventType.MEDIA:
                    audio_append = {
                        "type": "input_audio_buffer.append",
                        "audio": event.payload
                    }
                    await openai_websocket.send(json.dumps(audio_append))
                case StreamEventType.STREAM_STOPPED:
                    logger.info("stream stopped")
                    break
                case _:
                    logger.warning(f"Unhandled event type: {event.event_type}")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"WebSocket connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Error reading from Bandwidth WebSocket: {e}", exc_info=True)
    finally:
        if not bandwidth_websocket.client_state.name == "DISCONNECTED":
            try:
                await bandwidth_websocket.close()
            except Exception:
                pass
        try:
            await openai_websocket.close()
        except Exception:
            pass


async def receive_from_openai_ws(openai_websocket: ClientConnection, bandwidth_websocket: WebSocket, call_id: str):
    """
    Receive messages from OpenAI WebSocket and forward audio to Bandwidth WebSocket.
    :param openai_websocket:
    :param bandwidth_websocket:
    :param call_id:
    :return: None
    """
    last_assistant_item = None
    try:
        async for message in openai_websocket:
            openai_message = json.loads(message)
            message_type = openai_message.get('type')
            logger.debug(f"OpenAI message: {message_type}")
            
            match message_type:
                case 'session.created' | 'session.updated':
                    logger.info(f"OpenAI session event: {message_type}")
                case 'response.created' | 'response.done':
                    logger.debug(f"Response event: {message_type}")
                case 'response.output_audio.delta' if 'delta' in openai_message:
                    audio_payload = base64.b64encode(base64.b64decode(openai_message['delta'])).decode('utf-8')
                    media = StreamMedia(
                        content_type="audio/pcmu",
                        payload=audio_payload
                    )
                    play_audio_event = BandwidthStreamEvent(
                        event_type=StreamEventType.PLAY_AUDIO,
                        media=media
                    )
                    try:
                        await bandwidth_websocket.send_text(play_audio_event.model_dump_json(by_alias=True, exclude_none=True))
                    except Exception as e:
                        logger.warning(f"Failed to send audio to Bandwidth: {e}")
                        break
                case 'response.output_audio_transcript.done':
                    logger.info(openai_message.get('transcript'))
                case 'conversation.item.done':
                    if openai_message.get('item').get('type') == 'function_call':
                        function_name = openai_message.get('item').get('name')
                        handle_tool_call(function_name, call_id)
                case 'input_audio_buffer.speech_started':
                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": 0
                    }
                    await openai_websocket.send(json.dumps(truncate_event))
                    clear_event = BandwidthStreamEvent(
                        event_type=StreamEventType.CLEAR,
                    )
                    try:
                        await bandwidth_websocket.send_text(clear_event.model_dump_json(by_alias=True, exclude_none=True))
                    except Exception as e:
                        logger.warning(f"Failed to send clear event to Bandwidth: {e}")
                        break
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
        logger.error(f"OpenAI WebSocket connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Error reading from OpenAI WebSocket: {e}", exc_info=True)
    finally:
        try:
            await openai_websocket.close()
        except Exception:
            pass
        if not bandwidth_websocket.client_state.name == "DISCONNECTED":
            try:
                await bandwidth_websocket.close()
            except Exception:
                pass


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
    start_stream = StartStream(
        destination=f"{websocket_url}?call_id={call_id}",
        mode="bidirectional",
        name=call_id
    )
    bxml_response = Bxml(nested_verbs=[start_stream])
    
    bxml_content = bxml_response.to_bxml()
    logger.info(f"Sending BXML for call {call_id}: {bxml_content}")
    return Response(status_code=http.HTTPStatus.OK, content=bxml_content, media_type="application/xml")


@app.websocket("/ws")
async def handle_inbound_websocket(bandwidth_websocket: WebSocket, call_id: str = None):
    """
    Handle inbound WebSocket connections from Bandwidth and bridge to OpenAI WebSocket.
    :param bandwidth_websocket:
    :param call_id:
    :return: None
    """
    try:
        await bandwidth_websocket.accept()
        logger.info(f"Bandwidth WebSocket connection accepted for call ID: {call_id}")

        if not call_id:
            logger.error("No call_id provided in WebSocket connection")
            await bandwidth_websocket.close(code=1008, reason="Missing call_id parameter")
            return

        logger.info(f"Attempting to connect to OpenAI for call ID: {call_id}")
        try:
            # Add connection timeout to prevent hanging
            async with asyncio.timeout(10):
                async with websockets.connect(
                        f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
                        additional_headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "OpenAI-Beta": "realtime=v1"
                        },
                        open_timeout=10
                ) as openai_websocket:
                    logger.info(f"Connected to OpenAI WebSocket for call ID: {call_id}")
                    await initialize_openai_session(openai_websocket)
                    await asyncio.gather(
                        receive_from_bandwidth_ws(bandwidth_websocket, openai_websocket),
                        receive_from_openai_ws(openai_websocket, bandwidth_websocket, call_id)
                    )
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to OpenAI for call {call_id}")
            await bandwidth_websocket.close(code=1011, reason="OpenAI connection timeout")
            return
    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"OpenAI WebSocket connection failed with status {e.status_code}: {e}")
        try:
            await bandwidth_websocket.close(code=1011, reason="OpenAI connection failed")
        except:
            pass
    except Exception as e:
        logger.error(f"Error in WebSocket handler for call {call_id}: {e}", exc_info=True)
        try:
            await bandwidth_websocket.close(code=1011, reason="Internal server error")
        except:
            pass


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
    logger.info(f"Received disconnect event for call ID: {call_id}, cause: {disconnect_cause}, error: {error_message}")
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

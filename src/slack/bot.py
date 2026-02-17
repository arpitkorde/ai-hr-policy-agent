import logging
import asyncio
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from src.config import settings

logger = logging.getLogger(__name__)

# Initialize Slack Bolt App (Graceful failure if tokens are missing)
slack_app = None
app_handler = None

try:
    if settings.slack_bot_token and settings.slack_signing_secret:
        slack_app = AsyncApp(
            token=settings.slack_bot_token,
            signing_secret=settings.slack_signing_secret,
        )
        app_handler = AsyncSlackRequestHandler(slack_app)
        logger.info("Slack Bolt App initialized successfully.")
    else:
        logger.warning("Slack tokens not found in config. Slack Bot disabled.")
except Exception as e:
    logger.error(f"Failed to initialize Slack Bot: {e}")

# Global reference to the RAG chain
_rag_chain = None


def init_slack_bot(chain_instance):
    """Initialize the Slack bot with the RAG chain instance."""
    global _rag_chain
    _rag_chain = chain_instance
    if slack_app:
        logger.info("Slack Bot connected to RAG chain.")


async def start_socket_mode():
    """Start the Socket Mode handler in a background task."""
    if slack_app and settings.slack_app_token:
        try:
            handler = AsyncSocketModeHandler(slack_app, settings.slack_app_token)
            logger.info("Starting Slack Socket Mode handler...")
            await handler.start_async()
        except Exception as e:
            logger.error(f"Failed to start Socket Mode: {e}")
    elif not settings.slack_app_token:
        logger.warning("SLACK_APP_TOKEN not provided. Socket Mode disabled.")


# In-memory history storage: {thread_id: [(human, ai), ...]}
# Key: thread_ts (if threaded) or channel_id (if DM)
session_history = {}

def get_session_history(thread_id: str) -> list[tuple[str, str]]:
    """Retrieve chat history for a thread."""
    return session_history.get(thread_id, [])

def update_session_history(thread_id: str, human: str, ai: str):
    """Update chat history for a thread (keep last 5 turns)."""
    history = session_history.get(thread_id, [])
    history.append((human, ai))
    session_history[thread_id] = history[-5:]


if slack_app:
    @slack_app.event("app_mention")
    async def handle_app_mentions(event, say):
        """Handle @mentions in Slack channels."""
        user = event.get("user")
        text = event.get("text")
        channel = event.get("channel")
        thread_ts = event.get("ts")  # Reply in thread
        
        # Determine unique thread ID for history
        # Use thread_ts if it exists (replying in thread), otherwise use ts of the message -> NO
        # If user REPLIES to a thread, event["thread_ts"] exists.
        # If user STARTS a thread, event["thread_ts"] is missing, but we reply in thread using event["ts"].
        # So key should be event.get("thread_ts", event["ts"]).
        thread_id = event.get("thread_ts", event["ts"])

        logger.info(f"Received Slack mention from {user}: {text}")

        if not _rag_chain:
            await say("System is starting up, please try again in a moment.", thread_ts=thread_ts)
            return

        # Send a temporary "thinking" message
        wait_msg = await say(":thinking_face: Searching HR policies...", thread_ts=thread_ts)
        
        try:
            # Retrieve history
            history = get_session_history(thread_id)
            
            # Run RAG query in a separate thread
            # Pass history to the chain
            result = await asyncio.to_thread(_rag_chain.query, text, chat_history=history)

            # Update history with new interaction
            update_session_history(thread_id, text, result.answer)

            # Format the response using Slack Block Kit
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{result.answer}*"
                    }
                }
            ]

            # Add citations if available
            if result.sources:
                context_elements = []
                seen_sources = set()
                for source in result.sources:
                    doc = source.get("document", "Unknown")
                    page = source.get("page", "?")
                    src_str = f"{doc} (Page {page})"
                    if src_str not in seen_sources:
                        context_elements.append({
                            "type": "mrkdwn",
                            "text": f"📄 *{src_str}*"
                        })
                        seen_sources.add(src_str)
                
                if context_elements:
                    blocks.append({"type": "divider"})
                    blocks.append({
                        "type": "context",
                        "elements": context_elements
                    })

            # Update the "thinking" message with the answer
            await slack_app.client.chat_update(
                channel=channel,
                ts=wait_msg["ts"],
                text=result.answer, # Fallback text
                blocks=blocks
            )

        except Exception as e:
            logger.error(f"Slack processing error: {e}")
            await slack_app.client.chat_update(
                channel=channel,
                ts=wait_msg["ts"],
                text=f":warning: Sorry, I encountered an error: {str(e)}"
            )


    @slack_app.event("message")
    async def handle_message_events(event, say):
        """Handle direct messages (DMs)."""
        channel_type = event.get("channel_type")
        if channel_type != "im":
            return
        await handle_app_mentions(event, say)

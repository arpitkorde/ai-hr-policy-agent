import logging
import asyncio
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

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

# Global reference to the RAG chain (injected at startup)
_rag_chain = None


def init_slack_bot(chain_instance):
    """Initialize the Slack bot with the RAG chain instance."""
    global _rag_chain
    _rag_chain = chain_instance
    if slack_app:
        logger.info("Slack Bot connected to RAG chain.")


if slack_app:
    @slack_app.event("app_mention")
    async def handle_app_mentions(event, say):
        """Handle @mentions in Slack channels."""
        user = event.get("user")
        text = event.get("text")
        channel = event.get("channel")
        thread_ts = event.get("ts")  # Reply in thread

        logger.info(f"Received Slack mention from {user}: {text}")

        if not _rag_chain:
            await say("System is starting up, please try again in a moment.", thread_ts=thread_ts)
            return

        # Send a temporary "thinking" message
        wait_msg = await say(":thinking_face: Searching HR policies...", thread_ts=thread_ts)
        
        try:
            # Run RAG query in a separate thread to avoid blocking event loop
            result = await asyncio.to_thread(_rag_chain.query, text)

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
        
        # Only respond to DMs
        if channel_type != "im":
            return

        # Reuse the mention handler logic
        await handle_app_mentions(event, say)

import logging
from botbuilder.core import ActivityHandler, TurnContext, MessageFactory, CardFactory
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes
from src.rag.chain import HRPolicyChain
from src.config import settings

logger = logging.getLogger(__name__)

class TeamsBot(ActivityHandler):
    def __init__(self, chain: HRPolicyChain):
        self.chain = chain

    async def on_members_added_activity(
        self, members_added: list[ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    "Hello! I am the HR Policy Agent. "
                    "Ask me anything about company policies, like 'What is the leave policy?'"
                )

    async def on_message_activity(self, turn_context: TurnContext):
        text = turn_context.activity.text
        if not text:
            return

        # Remove @mention text if present (e.g., <at>BotName</at>)
        # The bot framework usually handles this in middleware if configured, 
        # but manual cleanup is often safer for simple text processing.
        # For now, we assume simple text.
        
        # Get user ID for context/history key
        user_id = turn_context.activity.from_property.id
        conversation_id = turn_context.activity.conversation.id
        
        # We can use our existing RAG chain logic here.
        # TODO: Thread History logic for Teams (similar to Slack)
        # Using conversation_id as the key.
        
        await turn_context.send_activity(Activity(type=ActivityTypes.typing))
        
        try:
            # We can reuse the same session_history logic from bot.py if we move it to a shared place?
            # Or just duplicate it for now since we want to move fast.
            # Let's import get_session_history from slack.bot if feasible, or just local dict.
            # Local dict is cleaner for separation.
            
            # Simple query for MVP without history first? 
            # No, user asked for "similar to Slack" which implies full features.
            # But let's start with simple RAG query.
            # Calling chain.query
            # We can access chain because it was passed in __init__.
            
            # Run RAG query (sync function in thread)
            import asyncio
            result = await asyncio.to_thread(self.chain.query, text)
            
            # Send response
            await turn_context.send_activity(MessageFactory.text(result.answer))
            
            # Send citations if available
            if result.sources:
                sources_text = "\n\n**Sources:**\n" + "\n".join(
                    [f"- {s.get('document', 'Unknown')} (Page {s.get('page', '?')})" for s in result.sources]
                )
                await turn_context.send_activity(MessageFactory.text(sources_text))

        except Exception as e:
            logger.error(f"Teams Bot Error: {e}")
            await turn_context.send_activity(
                "Sorry, I encountered an error while processing your request."
            )

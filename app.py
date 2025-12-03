import asyncio
import os
from typing import Any, List, Tuple

import gradio as gr

from sales_agent import handle_customer_message


def _format_response(agent_reply: Any) -> str:
    """Normalize various agent reply objects into plain text for display."""
    if hasattr(agent_reply, "content"):
        return agent_reply.content

    if isinstance(agent_reply, dict) and "content" in agent_reply:
        return str(agent_reply["content"])

    return str(agent_reply)


async def chat_endpoint(message: str, history):
    """Handle chat messages from Gradio interface."""
    try:
        # If this is the first message (empty history), return the greeting
        if not history:
            return "Welcome to Body Na MeatPie Skincare Store! How can I help you today?"
        
        # Pass the history directly - handle_customer_message will parse it
        # Gradio ChatInterface with type="messages" passes history as list of tuples
        reply = await handle_customer_message(message, conversation_history=history)
        return _format_response(reply)
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again."


def launch_interface():
    """Launch the Gradio chat interface."""
    demo = gr.ChatInterface(
        fn=chat_endpoint,
        type="messages",
        title="AI Sales Representative",
        description="Ask about our products, delivery, or pricing.",
        theme="soft",
    )
    
    # Check if share should be enabled (default: True)
    # Set ENABLE_SHARE=false in environment to disable
    enable_share = os.getenv("ENABLE_SHARE", "true").lower() == "true"
    
    if enable_share:
        print("Starting Gradio interface with public share link...")
        print("Note: Share link creation may take 10-30 seconds. Local URL will be available immediately.")
        try:
            # Launch with share=True
            # Setting server_name ensures local server starts even if share link creation is slow
            demo.launch(share=True, server_name="127.0.0.1", inbrowser=True)
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error with share link: {e}")
            print("Falling back to local-only mode...")
            demo.launch(server_name="127.0.0.1", inbrowser=True)
    else:
        print("Starting Gradio interface (local only)...")
        demo.launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    launch_interface()


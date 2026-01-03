import os
from pathlib import Path
import re
import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from knowledge import load_documents
from payment import create_payment
import time
from db import get_user, upsert_user
from fastapi import FastAPI, Request
from db import init_db

init_db()





BASE_DIR = Path(__file__).resolve().parent

# Load .env files so OPENAI_API_KEY and other secrets are available.
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

company_info = load_documents()

system_message = f"""
You are a professional AI and friendly sales assistant for a Nigerian skincare store.

Use only the company information below when answering:

{company_info}
 

IMPORTANT:
You NEVER generate payment links.
You NEVER guess payment links.
You NEVER say “payment link will be sent” unless one appears.

Your job is to:
- Recommend skincare products from the provided catalog only
- Explain benefits simply
- Collect customer details ONLY after they agree to buy

SALES FLOW YOU MUST FOLLOW:

1. Recommendation
- Ask skin type, concern, and budget ONLY if needed
- If customer asks for a specific product, explain it directly

2. Buying Intent
- When the customer says they want to buy:
  Ask for:
  - Full name
  - Phone number
  - Email address
  - Delivery address

3. Order Summary
- Clearly list:
  - Products
  - Total amount
- Ask:
  “Just to confirm, your order total is ₦X. Is this correct?”

4. Confirmation
- If the customer confirms:
  - Acknowledge politely
  - WAIT
  - Do NOT mention payment links
  - Do NOT invent placeholders

5. Payment
- If a payment link appears in your response, present it clearly
- After payment link:
  Say:
  “Once payment is confirmed, we’ll process your order.”

RULES:
- Never invent products
- Never invent prices
- Never invent links
- Never rush payment
"""


api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to ai_sales_rep/.env or the parent .env file."
    )

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

sales_agent = AssistantAgent(
    name="SalesAgent",
    system_message=system_message,
    model_client=model_client,
    model_client_stream=True,
)


async def handle_customer_message(chat_id: int, user_input: str):
    # 1️⃣ Load user from DB
    user = get_user(chat_id)

    state   = user[1] if user else "NEW"
    name    = user[2] if user else None
    phone   = user[3] if user else None
    email   = user[4] if user else None
    address = user[5] if user else None
    amount  = user[6] if user else None
    history = user[7] if user else ""

    # 2️⃣ Send message to AI (for recommendations only)
    messages = []
    if history:
        messages.append(TextMessage(content=history, source="user"))

    messages.append(TextMessage(content=user_input, source="user"))

    response = await sales_agent.on_messages(
        messages,
        cancellation_token=CancellationToken(),
    )

    agent_response = (
        response.chat_message.content
        if hasattr(response, "chat_message")
        else str(response)
    )

    # 3️⃣ Extract structured info from USER message
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
    if email_match:
        email = email_match.group(0)

    amount_match = extract_amount(history + " " + agent_response)
    if amount_match:
        amount = amount_match

    if "name" in user_input.lower():
        name = user_input.strip()

    if re.search(r'\+?\d{10,15}', user_input):
        phone = re.search(r'\+?\d{10,15}', user_input).group(0)

    if "address" in user_input.lower():
        address = user_input.strip()

    # 4️⃣ STATE MACHINE
    # -------------------------------------------------

    # 🔹 STEP A — Collect details
    if state in ["NEW", "COLLECTING_INFO"]:
        missing = []
        if not name: missing.append("full name")
        if not phone: missing.append("phone number")
        if not email: missing.append("email address")
        if not address: missing.append("delivery address")

        if missing:
            agent_response = (
                "To complete your order, I’ll need:\n"
                + "\n".join(f"- {m}" for m in missing)
            )
            state = "COLLECTING_INFO"
        else:
            agent_response = (
                "Got it! Here are your order details:\n\n"
                f"👤 Name: {name}\n"
                f"📞 Phone: {phone}\n"
                f"📧 Email: {email}\n"
                f"🏠 Address: {address}\n\n"
                f"💰 Total Amount: ₦{int(amount):,}\n\n"
                "Please confirm — is everything correct?"
            )
            state = "AWAITING_CONFIRMATION"

    # 🔹 STEP B — Await confirmation
    elif state == "AWAITING_CONFIRMATION":
        if detect_confirmation(user_input):
            payment_link = process_payment(email, amount)

            agent_response = (
                "Perfect! Your order is confirmed.\n\n"
                f"{payment_link}\n\n"
                "Once payment is confirmed, we will process your order."
            )
            state = "PAYMENT_SENT"
        else:
            agent_response = (
                "Please confirm if the order details and amount are correct "
                "so I can proceed."
            )

    # 🔹 STEP C — Payment already sent (LOCKED)
    elif state == "PAYMENT_SENT":
        agent_response = (
            "Your payment link has already been generated.\n\n"
            "Once payment is completed, we’ll process your order. 🙏"
        )

    # 5️⃣ Save back to DB
    updated_history = f"{history}\nUser: {user_input}\nBot: {agent_response}".strip()

    upsert_user(
        chat_id,
        state=state,
        name=name,
        phone=phone,
        email=email,
        address=address,
        amount=amount,
        history=updated_history
    )

    return agent_response





def extract_email(text: str):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None


def detect_confirmation(text: str):
    confirmations = [
        "yes", "correct", "confirm", "confirmed",
        "ok", "okay", "that's right", "proceed",
        "go ahead", "yes it is", "that's fine"
    ]
    text = text.lower()
    return any(word in text for word in confirmations)


def extract_amount(text):
    """
    Extract ONLY monetary values.
    Ignores phone numbers and random long digits.
    """
    if not text:
        return None

    
    total_match = re.search(
        r'(?:total(?: amount)?(?: is)?|amount(?: is)?)\s*[:\-]?\s*(?:₦|N)?\s*([\d,]+)',
        text,
        re.IGNORECASE
    )
    if total_match:
        return float(total_match.group(1).replace(",", ""))

    
    currency_matches = re.findall(
        r'(?:₦|N)\s*([\d,]{3,})',
        text
    )
    for amt in currency_matches:
        clean = amt.replace(",", "")
        if len(clean) <= 7:  
            return float(clean)

    return None




def process_payment(email, amount):
    """Process payment through Paystack and return payment link."""
    print("💳 process_payment called with:", email, amount)
    try:
        # Validate inputs
        if not email or not isinstance(email, str):
            return "Invalid email address provided."
        
        if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
            return "Invalid amount. Please provide a valid amount greater than 0."
        
        ref = f"order_{int(time.time())}"
        response = create_payment(email, amount, ref)

        
        if isinstance(response, dict):
            
            if response.get("status") is True:
                data = response.get("data", {})
                if isinstance(data, dict) and data.get("authorization_url"):
                    link = data["authorization_url"]
                    return f"Here is your secure payment link:\n{link}"
            
            # Handle error responses
            error_msg = response.get("message", "Unknown error occurred while generating payment link")
            return f"Something went wrong while generating your payment link: {error_msg}"
        else:
            return "Invalid response from payment service."
    except Exception as e:
        return f"Error processing payment: {str(e)}"
    

# FastAPI app instance for webhook endpoint
app = FastAPI()

@app.post("/paystack/webhook")
async def webhook(request: Request):
    body = await request.json()
    
    if body["event"] == "charge.success":
        email = body["data"]["customer"]["email"]
        amount = body["data"]["amount"] / 100

        print("Payment successful:", email, amount)

    return {"status": "ok"}


def send_telegram_message(chat_id, text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        print("❌ TELEGRAM_BOT_TOKEN missing")
        return

    if not text:
        text = "⚠️ Empty response from AI."

    # 🔴 Telegram message limit
    if len(text) > 4000:
        text = text[:4000]

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text
    }

    response = requests.post(url, json=payload)

    print("📨 TELEGRAM SEND STATUS:", response.status_code)
    print("📨 TELEGRAM SEND RESPONSE:", response.text)






@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    if "message" not in data:
        return {"ok": True}

    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")

    reply = await handle_customer_message(chat_id, text)
    send_telegram_message(chat_id, reply)

    return {"ok": True}


















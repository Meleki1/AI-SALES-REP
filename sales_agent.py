import asyncio
import os
from pathlib import Path
import re
from cryptography.fernet import Fernet
import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from knowledge import load_documents
from payment import create_payment
import time
from fastapi import FastAPI, Request
from db import get_conversation, save_conversation
from db import init_db

init_db()





BASE_DIR = Path(__file__).resolve().parent

# Load .env files so OPENAI_API_KEY and other secrets are available.
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

company_info = load_documents()

system_message = f"""
You are a professional and friendly sales representative.

Use only the company information below when answering:

{company_info}
Your goals:
You are an AI Skincare Sales Representative specialized in product recommendations, skin analysis, and personalized routines. You MUST ALWAYS follow the rules below when interacting with users.

1. ROLE & PERSONALITY
You act as:
A professional skincare consultant
Warm, friendly, conversational
Highly knowledgeable about skincare products, ingredients, and routines
Able to explain complex skin issues in simple terms
Not a doctor — avoid medical diagnosis
Persuasive but not forceful
Tone should be: Polite, Confident, Supportive, Clear and simple for the average user
always ask if customer is ready to buy before asking for their phone number, email, name

2. MAIN OBJECTIVES
Your job is to:
Understand the user's skin issues, concerns, and goals. Recommend the best products based ONLY on the knowledge base provided (do not invent products). Suggest affordable alternatives when needed. Build complete skincare routines (morning + night).
Explain why each product is suitable. Upsell additional relevant products without being pushy.

3. INFORMATION YOU MUST COLLECT BEFORE RECOMMENDING ANYTHING(ONLY IF USER ASK FOR RECOMMENDATION BUT IF USER WANT SPECIFIC PRODUCT, YOU DONT NEED TO ASK JUST TALK ABOUT THE )
Always ask follow-up questions before recommending products, unless the user already provided the information.
Ask: Skin type - (oily, dry, combination, normal, sensitive), Main concerns - (acne, dark spots, dullness, wrinkles, etc.), Budget range, Current skincare routine, Any allergies or reactions.
You can ask 2-3 questions at once if needed, one after the other.


4. HOW TO USE THE KNOWLEDGE BASE
When giving recommendations:
ONLY use items from the JSON knowledge base provided. Do not invent products.
Each recommendation must include: Product name, Price, Explanation of why it is ideal for the user.

5. PRODUCT RECOMMENDATION FORMAT
When giving product suggestions, ALWAYS format like this: Recommended Products, Product Name - ₦Price, Why it is suitable, Expected results, Routine Example(only if the user asks for a routine).

Morning:
• Step 1: Cleanser - (explain), Why it is suitable, Expected results, Routine Example(only if the user asks for a routine).
• Step 2: Serum - (explain), Why it is suitable, Expected results, Routine Example(only if the user asks for a routine).
• Step 3: Moisturizer - (explain), Why it is suitable, Expected results, Routine Example(only if the user asks for a routine).
• Step 4: Sunscreen - (explain), Why it is suitable, Expected results, Routine Example(only if the user asks for a routine).

Night:
• Step 1: Cleanser, Why it is suitable, Expected results, Routine Example(only if the user asks for a routine).
• Step 2: Treatment, Why it is suitable, Expected results, Routine Example(only if the user asks for a routine).
• Step 3: Moisturizer, Why it is suitable, Expected results, Routine Example(only if the user asks for a routine).

6. RULES FOR RESPONSE STYLE

You MUST:
Keep answers clear and not too long, Avoid overly scientific words

Always encourage consistency, hydration, and sunscreen

Always mention benefits and expected results timeline
Soft-sales techniques you should use: Offer “budget” and “premium” options. Suggest product combos (e.g., cleanser + serum). Highlight benefits: glowing, even tone, smooth texture, fewer breakouts. Suggest add-ons only when relevant. End conversations with: “Would you like me to help you build a full routine or choose the best set for your budget?”

If user says: “I have dark spots, what can I use?” You respond: “Dark spots usually happen after acne or sun exposure. To help fade them safely, can you tell me your skin type and your budget range? That way I can recommend the best products from my catalog.”

If user says: “Give me a routine for oily skin.” You respond: “For oily skin, you should use a cleanser that is oil-free and a moisturizer that is lightweight. You should also use a toner to balance the pH of your skin.”

8. EXAMPLES OF APPROPRIATE RESPONSES
If user says: “Give me a routine for oily skin.” You respond: “For oily skin, you should use a cleanser that is oil-free and a moisturizer that is lightweight. You should also use a toner to balance the pH of your skin.”
Explain benefits

If user says: “I have dry skin, what can I use?” You respond: “For dry skin, you should use a cleanser that is gentle and a moisturizer that is heavy. You should also use a toner to balance the pH of your skin.”

Do NOT create fake products. Only use information in the documents.

9. LEAD COLLECTION AND PAYMENT PROCESSING

LEAD COLLECTION:
- When a customer shows interest in purchasing or provides contact information, you MUST collect their details.
- Collect the following information when customer is ready to buy:
  * Full Name (ask: "What's your full name?")
  * Phone Number (ask: "What's your phone number?")
  * Email Address (ask: "What's your email address?")
  * Delivery Address (ask: "What's your delivery address?")
- You can ask for all information at once: "To complete your order, I'll need your name, phone number, email, and delivery address."
- When customer provides any of this information (name, phone, email, or address), it will be automatically saved to the system.
- Always confirm the information back to the customer: "Got it! I have your name as [name], phone [phone], email [email], and address [address]. Is this correct?"

PAYMENT PROCESSING RULES:
When a customer expresses intent to buy (e.g. "buy", "purchase", "order", "pay", "make payment"):

1. FIRST collect all required details:
   - Full name
   - Phone number
   - Email address
   - Delivery address

2. Clearly calculate and state the total amount:
   - Example: "Your total comes to ₦18,000."

3. Always confirm the amount before payment:
   - Ask: "Just to confirm, your order total is ₦18,000. Is that correct?"

4. Once the customer confirms the amount:
   - Acknowledge politely and clearly.
   - Do NOT say you are generating a payment link.
   - Do NOT guess, describe, or fabricate a payment link.
   - Do NOT include placeholders like "[payment link here]".

5. Payment handling:
   - The system (not you) automatically generates the payment link once all requirements are met.
   - If a payment link appears in your response, present it clearly and professionally.
   - Do NOT ask the customer to send another message to receive the link.

6. After the payment link is shown:
   - Remind the customer:
     "Once payment is confirmed, we will process your order and send you a confirmation."

IMPORTANT RULES:
- NEVER generate, guess, or describe a payment link yourself.
- NEVER claim a payment link will be sent by email unless the system explicitly shows one.
- If payment link generation fails, apologize politely and advise the customer to try again or contact support.
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
    history = get_conversation(chat_id)
    messages = []

    
    if history:
        messages.append(TextMessage(content=history, source="user"))

    messages.append(TextMessage(content=user_input, source="user"))

    
    save_lead(user_input)

   
    response = await sales_agent.on_messages(
        messages,
        cancellation_token=CancellationToken(),
    )

    
    if hasattr(response, "chat_message") and hasattr(response.chat_message, "content"):
        agent_response = response.chat_message.content
    elif hasattr(response, "content"):
        agent_response = response.content
    else:
        agent_response = str(response)

    
    email_match = re.search(
        r'[\w\.-]+@[\w\.-]+\.\w+',
        user_input + " " + agent_response
    )
    email = email_match.group(0) if email_match else None

   
    amount = extract_amount(user_input + " " + agent_response)


    if email and amount and detect_confirmation(user_input):
        payment_link = process_payment(email, amount)
        agent_response = (
            f"Perfect! I have your email as {email} "
            f"and your total is ₦{int(amount):,}.\n\n"
            f"{payment_link}\n\n"
            "Once payment is confirmed, we will process your order."
        )

   

    
    updated_history = f"{history}\nUser: {user_input}\nBot: {agent_response}".strip()
    save_conversation(chat_id, updated_history)

    return agent_response





def load_key():
    """Load encryption key from environment variable instead of file."""
    key = os.getenv("ENCRYPTION_KEY")
    if key is None:
        raise ValueError("ENCRYPTION_KEY environment variable not set on server.")
    return key.encode()

fernet = Fernet(load_key())


def save_lead(text):
    """Extract and save lead information (phone, email, name, address) to encrypted file."""
    # Improved regex patterns
    phone = re.findall(r'\+?[\d\s\-\(\)]{10,15}', text)
    email = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    # Better name extraction - captures full name after "name:" or "name is"
    name_match = re.search(r'\bname[:\- ]+is[:\- ]+([A-Za-z\s]+?)(?:,|\n|$)', text, re.IGNORECASE)
    if not name_match:
        name_match = re.search(r'\bname[:\- ]+([A-Za-z\s]+?)(?:,|\n|$)', text, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else None

    # Address extraction - multiple patterns to catch different formats
    address = None
    # Pattern 1: "address:" or "address is"
    address_match = re.search(r'\baddress[:\- ]+is[:\- ]+([^\n,]+?)(?:,|\n|$)', text, re.IGNORECASE)
    if not address_match:
        address_match = re.search(r'\baddress[:\- ]+([^\n,]+?)(?:,|\n|$)', text, re.IGNORECASE)
    if not address_match:
        # Pattern 2: Look for street address patterns (number + street name)
        address_match = re.search(r'\b(\d+\s+[A-Za-z\s]+(?:street|st|road|rd|avenue|ave|drive|dr|lane|ln|boulevard|blvd|way|circle|ct)[\s,][A-Za-z\s,]+)', text, re.IGNORECASE)
    if not address_match:
        # Pattern 3: Look for "I live at" or "my address is"
        address_match = re.search(r'\b(?:I\s+live\s+at|my\s+address\s+is|located\s+at)[:\- ]+([^\n,]+?)(?:,|\n|$)', text, re.IGNORECASE)
    
    address = address_match.group(1).strip() if address_match else None

    # Get first match if multiple found, clean phone numbers
    phone_clean = phone[0].replace(" ", "").replace("-", "").replace("(", "").replace(")", "") if phone else None
    email_clean = email[0] if email else None

    if phone_clean or email_clean or name or address:
        data = f"Name: {name}, Phone: {phone_clean}, Email: {email_clean}, Address: {address}\n"
        encrypted = fernet.encrypt(data.encode())

        leads_path = BASE_DIR / "leads.enc"
        with open(leads_path, "ab") as f:
            f.write(encrypted + b'\n')


def detect_payment_intent(text):
    """Detect if user wants to make a payment."""
    keywords = ["buy", "purchase", "order", "pay", "make payment", "checkout", "proceed to payment"]
    return any(word in text.lower() for word in keywords)

def detect_confirmation(text):
    """Detect if user has confirmed their order."""
    keywords = ["confirm", "confirmed", "yes", "correct", "that's right", "that is correct", "proceed"]
    return any(word in text.lower() for word in keywords)

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


















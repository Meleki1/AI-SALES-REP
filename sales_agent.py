import asyncio
import os
from pathlib import Path
import re
from cryptography.fernet import Fernet
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from knowledge import load_documents
from payment import create_payment
import time
from fastapi import FastAPI, Request


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

3. INFORMATION YOU MUST COLLECT BEFORE RECOMMENDING ANYTHING
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

PAYMENT PROCESSING:
- When customer expresses intent to buy (uses words like "buy", "purchase", "order", "pay", "make payment"), you should:
  1. First, collect all necessary information (name, phone, email, address)
  2. Calculate the total amount for the products they want
  3. Clearly state the total amount: "Your total comes to ₦[amount]"
  4. Ask for their email address if not already collected: "What's your email address for the payment link?"
  5. Once you have both the email and total amount, mention both in your response:
     Example: "Perfect! I have your email as [email] and your total is ₦[amount]. Let me generate your payment link..."
  6. The system will automatically detect the email and amount from your response and generate the payment link
  7. The payment link will appear in your response automatically
- Always confirm the amount before processing: "Just to confirm, your order total is ₦[amount]. Is that correct?"
- After the payment link is generated, remind them: "Once payment is confirmed, we'll process your order and send you a confirmation."
- Make sure to mention both the email and amount clearly in the same response when ready to process payment

IMPORTANT NOTES:
- Always ask if customer is ready to buy BEFORE asking for contact details
- Be natural and conversational when collecting information - don't make it feel like an interrogation
- If customer provides information in a single message (e.g., "My name is John, email is john@email.com, phone is 1234567890"), acknowledge all of it
- When customer says they want to buy, immediately proceed to collect information and process payment
- If payment link generation fails, apologize and ask them to try again or contact support
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


async def handle_customer_message(user_input: str, conversation_history=None):
    """
    Send a customer message to the sales agent with conversation history.
    Automatically saves leads and processes payments when detected.
    
    Args:
        user_input: The current user message
        conversation_history: List of tuples (user_msg, agent_msg) from previous conversation
    """
    # Build message list from conversation history
    messages = []
    
    if conversation_history:
        # Convert Gradio history format to autogen messages
        for item in conversation_history:
            # Handle different history formats
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    user_msg = item[0] if item[0] else ""
                    agent_msg = item[1] if item[1] else ""
                else:
                    continue
            else:
                continue
            
            # Add user message if it exists
            if user_msg:
                messages.append(TextMessage(content=str(user_msg), source="user"))
            # Add agent response if it exists
            if agent_msg:
                messages.append(TextMessage(content=str(agent_msg), source="SalesAgent"))
    
    # Add the current user message
    messages.append(TextMessage(content=user_input, source="user"))
    
    # Automatically save lead information if customer provides contact details
    # This extracts name, phone, email, and address from the user's message
    save_lead(user_input)
    
    # Send all messages to the agent (this gives it full conversation context)
    response = await sales_agent.on_messages(
        messages,
        cancellation_token=CancellationToken(),
    )
    
    # Extract just the text content from the response
    if hasattr(response, 'chat_message') and hasattr(response.chat_message, 'content'):
        agent_response = response.chat_message.content
    elif hasattr(response, 'content'):
        agent_response = response.content
    else:
        agent_response = str(response)
    
    # Check if agent response contains payment request format and process it
    payment_match = re.search(r'PAYMENT_REQUEST:\s*email=([\w\.-]+@[\w\.-]+\.\w+),\s*amount=([\d.]+)', agent_response, re.IGNORECASE)
    if payment_match:
        email = payment_match.group(1)
        amount = float(payment_match.group(2))
        payment_link = process_payment(email, amount)
        # Replace the placeholder with actual payment link
        agent_response = agent_response.replace(
            payment_match.group(0),
            payment_link
        )
    # Check if agent response contains both email and amount (agent is ready to process payment)
    else:
        # Build full conversation text for extraction
        full_conversation = user_input + " " + agent_response
        if conversation_history:
            for item in conversation_history:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    full_conversation += " " + str(item[0]) + " " + str(item[1])
        
        # Extract email and amount from full conversation
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', full_conversation, re.IGNORECASE)
        amount = extract_amount(full_conversation)
        
        # Process payment if we have both email and amount, and payment intent is detected
        if email_match and amount and (detect_payment_intent(user_input) or detect_payment_intent(agent_response)):
            email = email_match.group(0)
            # Only process if payment link not already in response
            if "payment link" not in agent_response.lower() and "authorization_url" not in agent_response.lower():
                payment_link = process_payment(email, amount)
                agent_response += f"\n\n{payment_link}"
    
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

def extract_amount(text):
    """Extract payment amount from text (looks for currency symbols and numbers)."""
    # Pattern for amounts like ₦5000, N5000, 5000, $50, etc.
    amount_patterns = [
        r'[₦N$]\s*(\d+(?:[.,]\d+)?)',  # Currency symbol followed by number
        r'(\d+(?:[.,]\d+)?)\s*(?:naira|ngn|usd|dollars?)',  # Number followed by currency word
        r'total[:\- ]+[₦N$]?\s*(\d+(?:[.,]\d+)?)',  # "total: ₦5000"
        r'amount[:\- ]+[₦N$]?\s*(\d+(?:[.,]\d+)?)',  # "amount: ₦5000"
    ]
    
    for pattern in amount_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                return float(amount_str)
            except ValueError:
                continue
    return None

def process_payment(email, amount):
    """Process payment through Paystack and return payment link."""
    try:
        ref = f"order_{int(time.time())}"
        response = create_payment(email, amount, ref)

        # Handle different response structures
        if isinstance(response, dict):
            if response.get("status") and response.get("data", {}).get("authorization_url"):
                link = response["data"]["authorization_url"]
                return f"Here is your secure payment link:\n{link}"
            else:
                error_msg = response.get("message", "Unknown error")
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
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text
    }

    requests.post(url, json=payload)





@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    if "message" not in data:
        return {"ok": True}

    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")

    reply = await handle_customer_message(text)
    send_telegram_message(chat_id, reply)

    return {"ok": True}













if __name__ == "__main__":
    customer_text = input("Customer message: ").strip()

    if not customer_text:
        raise ValueError("Customer message cannot be empty.")

    reply = asyncio.run(handle_customer_message(customer_text))
    print(reply)





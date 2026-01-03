# payment.py
import os
import time
from paystackapi.transaction import Transaction

PAYSTACK_SECRET = os.getenv("PAYSTACK_SECRET_KEY")

def generate_payment_link(email, amount):
    if not PAYSTACK_SECRET:
        return None, "Paystack key not set"

    ref = f"order_{int(time.time())}"

    try:
        response = Transaction.initialize(
            secret_key=PAYSTACK_SECRET,
            email=email,
            amount=int(amount * 100),
            reference=ref
        )

        if response.get("status") and response.get("data"):
            return response["data"]["authorization_url"], None

        return None, response.get("message", "Payment failed")

    except Exception as e:
        return None, str(e)

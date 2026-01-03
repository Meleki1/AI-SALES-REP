
import os
import time
from paystackapi.transaction import Transaction

PAYSTACK_SECRET = os.getenv("PAYSTACK_SECRET_KEY")

def create_payment(email, amount, reference=None):
    if not PAYSTACK_SECRET:
        return {
            "status": False,
            "message": "Paystack key not set"
        }

    if not reference:
        reference = f"order_{int(time.time())}"

    try:
        response = Transaction.initialize(
            secret_key=PAYSTACK_SECRET,
            email=email,
            amount=int(amount * 100),
            reference=reference
        )

        if response.get("status") and response.get("data"):
            return {
                "status": True,
                "data": response["data"]
            }

        return {
            "status": False,
            "message": response.get("message", "Payment failed")
        }

    except Exception as e:
        return {
            "status": False,
            "message": str(e)
        }


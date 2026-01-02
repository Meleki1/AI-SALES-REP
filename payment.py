import os
from paystackapi.paystack import Paystack
from paystackapi.transaction import Transaction

# Initialize Paystack only if secret key is available
_paystack_secret = os.getenv("PAYSTACK_SECRET_KEY")
paystack = Paystack(secret_key=_paystack_secret) if _paystack_secret else None

def create_payment(email, amount, reference):
    """Create a payment initialization request through Paystack and return payment link."""
    if not _paystack_secret:
        return {
            "status": False,
            "message": "PAYSTACK_SECRET_KEY is not set in environment variables"
        }

    try:
        response = Transaction.initialize(
            secret_key=_paystack_secret,
            email=email,
            amount=int(amount * 100),
            reference=reference
        )

        # Expect Paystack standard response
        if isinstance(response, dict) and response.get("status") is True:
            return response

        return {
            "status": False,
            "message": response.get("message", "Paystack initialization failed")
        }

    except Exception as e:
        return {
            "status": False,
            "message": f"Payment initialization failed: {str(e)}"
        }

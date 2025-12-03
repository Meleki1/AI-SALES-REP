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
        # Transaction.initialize() requires the secret key to be passed
        # The paystackapi library reads from environment or we pass it explicitly
        response = Transaction.initialize(
            secret_key=_paystack_secret,
            email=email,
            amount=int(amount * 100),  # Paystack uses kobo (smallest currency unit)
            reference=reference
        )
        
        # Verify response structure
        if isinstance(response, dict):
            # Check if response has the expected structure
            if response.get("status") and response.get("data"):
                return response
            else:
                # Some API versions return data directly
                return {
                    "status": True,
                    "data": response if isinstance(response, dict) else {"authorization_url": str(response)}
                }
        return response
    except Exception as e:
        return {
            "status": False,
            "message": f"Payment initialization failed: {str(e)}"
        }

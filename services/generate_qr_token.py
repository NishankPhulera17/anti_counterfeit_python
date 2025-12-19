import jwt
import datetime

SECRET_KEY = "nishankphulera"

def generate_qr_token(product_id):
    payload = {
        "product_id": product_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=10),  # token valid for 10 mins
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    # Ensure token is a string (PyJWT 2.0+ returns string, but older versions return bytes)
    if isinstance(token, bytes):
        return token.decode('utf-8')
    return str(token)
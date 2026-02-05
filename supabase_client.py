import os
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi import Header, HTTPException
import jwt

load_dotenv()

url = os.environ.get("SUPABASE_URL", "")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
jwt_secret = os.environ.get("SUPABASE_JWT_SECRET", "")

if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

clean_key = key.strip().replace("\n", "").replace("\r", "")

if not jwt_secret:
    print("WARNING: SUPABASE_JWT_SECRET is not set. Token verification will fail.")

supabase: Client = create_client(url, clean_key)

async def get_current_user(authorization: str = Header(None)):
    ANONYMOUS_USER_ID = "00000000-0000-0000-0000-000000000000"
    
    if not authorization or not authorization.startswith("Bearer "):
        return {"sub": ANONYMOUS_USER_ID, "email": "anonymous@example.com"}
    
    token = authorization.split(" ")[1]
    if token == "undefined" or token == "null":
        return {"sub": ANONYMOUS_USER_ID, "email": "anonymous@example.com"}
    
    try:
        pem_secret = jwt_secret
        if pem_secret:
            pem_secret = pem_secret.strip('"').strip("'")
            pem_secret = pem_secret.replace("\\n", "\n")
            if "BEGIN PUBLIC KEY" in pem_secret:
                lines = [line.strip() for line in pem_secret.split("\n") if line.strip()]
                header = "-----BEGIN PUBLIC KEY-----"
                footer = "-----END PUBLIC KEY-----"
                body_lines = [l for l in lines if "-----" not in l]
                pem_secret = f"{header}\n" + "\n".join(body_lines) + f"\n{footer}"

        try:
            return jwt.decode(token, pem_secret, algorithms=["HS256", "ES256"], options={"verify_aud": False})
        except Exception as e:
            raise e
    except jwt.ExpiredSignatureError:
        print("DEBUG AUTH: Token expirado")
        raise HTTPException(status_code=401, detail="Token has expired")
    except Exception as e:
        print(f"DEBUG AUTH: Error final de validación: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Auth error: {str(e)}")

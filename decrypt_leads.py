from cryptography.fernet import Fernet

def load_key():
    with open("secret.key", "rb") as f:
        return f.read()

fernet = Fernet(load_key())

with open("leads.enc", "rb") as f:
    for line in f:
        decrypted = fernet.decrypt(line.strip())
        print(decrypted.decode())

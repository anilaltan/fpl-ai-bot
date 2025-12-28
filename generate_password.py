#!/usr/bin/env python3
"""
Password Hash Generator for FPL AI Bot
Usage: python generate_password.py
"""

import bcrypt
import getpass

def generate_hashed_password(password: str) -> str:
    """
    Generate a bcrypt hashed password
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    # Convert password to bytes
    password_bytes = password.encode('utf-8')
    
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Return as string
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against its hash
    
    Args:
        password: Plain text password
        hashed: Hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    password_bytes = password.encode('utf-8')
    hashed_bytes = hashed.encode('utf-8')
    
    return bcrypt.checkpw(password_bytes, hashed_bytes)

def main():
    """Main function to generate password hashes"""
    print("=" * 60)
    print("FPL AI Bot - Password Hash Generator")
    print("=" * 60)
    print()
    
    # Get password from user
    password = getpass.getpass("Enter password to hash: ")
    
    if not password:
        print("❌ Error: Password cannot be empty!")
        return
    
    # Confirm password
    confirm = getpass.getpass("Confirm password: ")
    
    if password != confirm:
        print("❌ Error: Passwords do not match!")
        return
    
    print("\n🔄 Generating hash...")
    
    # Generate hash
    hashed = generate_hashed_password(password)
    
    print("\n✅ Password hashed successfully!")
    print("\n" + "=" * 60)
    print("HASHED PASSWORD (copy this to config.yaml):")
    print("=" * 60)
    print(hashed)
    print("=" * 60)
    
    # Verify the hash works
    print("\n🔍 Verifying hash...")
    if verify_password(password, hashed):
        print("✅ Verification successful!")
    else:
        print("❌ Verification failed!")
    
    print("\n📝 To use this password:")
    print("1. Copy the hashed password above")
    print("2. Add/update user in config.yaml")
    print("3. Paste the hashed password in the 'password' field")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

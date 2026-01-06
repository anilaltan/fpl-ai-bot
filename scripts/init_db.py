#!/usr/bin/env python3
"""
Database Initialization Script for FPL SaaS Application

This script creates all database tables and initializes the admin user.
"""

import sys
import os
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.orm import Session
from sqlalchemy import text
import bcrypt

from src.database import engine, SessionLocal, Base
from src.models import User


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def create_admin_user(db: Session) -> None:
    """
    Create admin user if it doesn't exist.

    Args:
        db: Database session
    """
    # Check if admin user already exists
    admin_user = db.query(User).filter(User.username == "admin").first()

    if admin_user:
        print("ℹ️  Admin kullanıcısı zaten mevcut.")
        return

    # Create admin user
    hashed_password = hash_password("admin123")
    admin = User(
        username="admin",
        hashed_password=hashed_password,
        role="admin"
    )

    db.add(admin)
    db.commit()
    db.refresh(admin)

    print("✅ Admin kullanıcısı oluşturuldu:")
    print(f"   Username: admin")
    print(f"   Password: admin123")
    print(f"   Role: admin")


def main():
    """Main initialization function."""
    print("🚀 FPL SaaS Database Initialization")
    print("=" * 50)

    try:
        # Drop existing players table if it exists (for schema updates)
        print("📦 Eski tablolar temizleniyor...")
        with engine.connect() as conn:
            # Drop players table if exists
            conn.execute(text("DROP TABLE IF EXISTS players CASCADE"))
            conn.commit()
        print("✅ Eski tablolar temizlendi.")

        # Create all tables
        print("📦 Yeni tablolar oluşturuluyor...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tablolar başarıyla oluşturuldu.")

        # Create admin user
        print("\n👤 Admin kullanıcısı kontrol ediliyor...")
        db = SessionLocal()
        try:
            create_admin_user(db)
        finally:
            db.close()

        print("\n🎉 Veritabanı başarıyla ilklendirildi!")
        print("✅ Tablolar ve Admin kullanıcısı oluşturuldu.")

    except Exception as e:
        print(f"\n❌ Hata: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

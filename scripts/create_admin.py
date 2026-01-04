import sqlite3
import bcrypt

# Veritabanı dosyası (Yolun doğruluğundan emin ol)
DB_PATH = "/root/fpl-test/fpl_saas.db"

def create_admin():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Tablo yoksa oluştur (Garanti olsun)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            fpl_id TEXT,
            subscription_plan TEXT DEFAULT 'free',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Şifreyi hashle
        password = "admin123"
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Kullanıcıyı ekle veya güncelle
        cursor.execute('''
        INSERT INTO users (username, email, password_hash, subscription_plan)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            password_hash = excluded.password_hash,
            subscription_plan = 'pro'
        ''', ('admin', 'admin@fpl.com', hashed, 'pro'))

        conn.commit()
        print("✅ BAŞARILI: 'admin' kullanıcısı oluşturuldu/güncellendi.")
        print("➡️ Kullanıcı: admin")
        print("➡️ Şifre: admin123")

    except Exception as e:
        print(f"❌ HATA: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_admin()
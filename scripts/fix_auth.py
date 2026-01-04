import sqlite3
import bcrypt
import os

# Backend'in baktığı muhtemel yollar
DB_PATHS = [
    "/root/fpl-test/fpl_saas.db",
    "fpl_saas.db"
]

def fix_admin():
    db_found = False
    target_db = None

    # 1. Doğru Veritabanını Bul
    for path in DB_PATHS:
        if os.path.exists(path):
            print(f"📂 Veritabanı bulundu: {path}")
            target_db = path
            db_found = True
            break
    
    if not db_found:
        print("❌ HATA: Veritabanı dosyası (fpl_saas.db) bulunamadı!")
        return

    conn = sqlite3.connect(target_db)
    cursor = conn.cursor()

    # 2. Mevcut Kullanıcıları Listele
    print("\n--- MEVCUT KULLANICILAR ---")
    try:
        cursor.execute("SELECT id, username, subscription_plan FROM users")
        users = cursor.fetchall()
        for u in users:
            print(f"ID: {u[0]} | User: {u[1]} | Plan: {u[2]}")
        
        if not users:
            print("⚠️ Tablo boş! Kullanıcı yok.")
    except Exception as e:
        print(f"❌ Tablo okuma hatası: {e}")
        conn.close()
        return

    # 3. Admin Şifresini Sıfırla (Bcrypt ile)
    print("\n--- ŞİFRE SIFIRLAMA ---")
    try:
        new_pass = "admin123"
        # Backend ile uyumlu hash üret
        hashed = bcrypt.hashpw(new_pass.encode('utf-8'), bcrypt.gensalt())
        
        # Kullanıcıyı güncelle veya ekle
        cursor.execute("SELECT * FROM users WHERE username='admin'")
        if cursor.fetchone():
            print("🔄 Admin kullanıcısı güncelleniyor...")
            cursor.execute("UPDATE users SET password_hash=?, subscription_plan='pro' WHERE username='admin'", (hashed,))
        else:
            print("➕ Admin kullanıcısı yeniden oluşturuluyor...")
            cursor.execute("INSERT INTO users (username, email, password_hash, subscription_plan) VALUES (?, ?, ?, ?)", 
                           ('admin', 'admin@test.com', hashed, 'pro'))
        
        conn.commit()
        print(f"✅ BAŞARILI: 'admin' şifresi '{new_pass}' olarak ayarlandı.")
        
        # 4. Doğrulama Testi
        cursor.execute("SELECT password_hash FROM users WHERE username='admin'")
        stored_hash = cursor.fetchone()[0]
        
        # Hash text mi byte mı kontrol et (SQLite bazen text saklar)
        if isinstance(stored_hash, str):
            stored_hash = stored_hash.encode('utf-8')
            
        if bcrypt.checkpw(new_pass.encode('utf-8'), stored_hash):
            print("✅ TEST GEÇTİ: Hash doğrulama başarılı (Local).")
        else:
            print("❌ TEST KALDI: Hash doğrulama başarısız!")

    except Exception as e:
        print(f"❌ İşlem hatası: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    fix_admin()
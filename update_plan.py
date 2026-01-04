#!/usr/bin/env python3
"""
FPL SaaS - Kullanıcı Plan Güncelleme Script'i
"""

import sqlite3
import sys

def update_user_plan(username, new_plan):
    """Kullanıcının planını güncelle"""
    try:
        conn = sqlite3.connect('fpl_saas.db')
        cursor = conn.cursor()
        
        # Kullanıcıyı kontrol et
        cursor.execute('SELECT id, username, subscription_plan FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if not user:
            print(f"❌ Kullanıcı '{username}' bulunamadı")
            return False
            
        old_plan = user[2]
        print(f"📋 {username} kullanıcısının planı {old_plan} → {new_plan} olarak güncelleniyor...")
        
        # Planı güncelle
        cursor.execute('UPDATE users SET subscription_plan = ? WHERE username = ?', (new_plan, username))
        conn.commit()
        
        print(f"✅ {username} kullanıcısının planı başarıyla güncellendi!")
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False
    finally:
        conn.close()

def list_users():
    """Tüm kullanıcıları listele"""
    try:
        conn = sqlite3.connect('fpl_saas.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT username, subscription_plan FROM users')
        users = cursor.fetchall()
        
        print("👥 Mevcut Kullanıcılar:")
        for username, plan in users:
            print(f"  • {username}: {plan}")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("🎯 FPL SaaS Kullanıcı Plan Güncelleme")
        print("Kullanım:")
        print("  python3 update_plan.py list                    # Kullanıcıları listele")
        print("  python3 update_plan.py <username> <plan>      # Planı güncelle")
        print()
        print("Örnekler:")
        print("  python3 update_plan.py admin premium")
        print("  python3 update_plan.py demo free")
        print()
        list_users()
        
    elif len(sys.argv) == 2 and sys.argv[1] == 'list':
        list_users()
        
    elif len(sys.argv) == 3:
        username = sys.argv[1]
        new_plan = sys.argv[2]
        update_user_plan(username, new_plan)
        
    else:
        print("❌ Geçersiz kullanım. Yardım için: python3 update_plan.py")

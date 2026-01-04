#!/usr/bin/env python3
"""
FPL SaaS Database Viewer
Database içeriğini görüntülemek için basit bir script
"""

import sqlite3
import pandas as pd
from pathlib import Path

def view_database(db_path='fpl_saas.db'):
    """Database içeriğini göster"""
    if not Path(db_path).exists():
        print(f"❌ Database dosyası bulunamadı: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        print('🎯 FPL SaaS Database Görüntüleyici')
        print('=' * 40)
        
        # Tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall() if table[0] != 'sqlite_sequence']
        
        print(f'📊 Bulunan tablolar: {", ".join(tables)}')
        print()
        
        # Users table
        if 'users' in tables:
            print('👥 Kullanıcılar:')
            print('-' * 15)
            df = pd.read_sql_query('''
                SELECT 
                    id,
                    username,
                    email,
                    CASE WHEN fpl_id IS NULL THEN 'Belirtilmemiş' ELSE fpl_id END as fpl_id,
                    subscription_plan,
                    created_at
                FROM users
            ''', conn)
            print(df.to_string(index=False))
            
        conn.close()
        print()
        print('✅ Database başarıyla görüntülendi!')
        
    except Exception as e:
        print(f'❌ Hata: {e}')

if __name__ == '__main__':
    view_database()

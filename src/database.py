"""
FPL SaaS Database Module

This module contains the DatabaseManager class for user authentication and data persistence.
Uses SQLite for database and bcrypt for password hashing.
"""

import logging
import sqlite3
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import bcrypt

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager for FPL SaaS application.

    Handles user authentication, registration, and FPL ID management.
    """

    def __init__(self, db_path: str = "/root/fpl-test/fpl_saas.db"):
        """
        Initialize database connection and create tables if they don't exist.

        Args:
            db_path: Path to SQLite database file
        """
        # FORCE ABSOLUTE PATH to avoid path confusion
        self.db_path = Path("/root/fpl-test/fpl_saas.db")
        print(f"🔥 DATABASE PATH ZORLANDI: {self.db_path}")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create users table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    fpl_id TEXT,
                    subscription_plan TEXT DEFAULT 'free',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.info("Database tables created/verified")

    def create_user(self, username: str, email: str, password: str) -> bool:
        """
        Create a new user with hashed password.

        Args:
            username: Unique username
            email: User's email address
            password: Plain text password (will be hashed)

        Returns:
            bool: True if user created successfully, False if username exists
        """
        try:
            # Hash the password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (username, email, password_hash)
                    VALUES (?, ?, ?)
                """, (username, email, password_hash))
                conn.commit()

            logger.info(f"User {username} created successfully")
            return True

        except sqlite3.IntegrityError:
            logger.warning(f"Username {username} already exists")
            return False
        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            return False

    def verify_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Verify user credentials and return user data if valid.

        Args:
            username: Username to verify
            password: Plain text password

        Returns:
            Optional[Dict]: User data dict if valid, None if invalid
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, username, email, password_hash, fpl_id, subscription_plan
                    FROM users
                    WHERE username = ?
                """, (username,))

                user_row = cursor.fetchone()

                if not user_row:
                    print(f"DEBUG AUTH: User {username} not found in database")
                    logger.warning(f"User {username} not found")
                    return None

                print(f"DEBUG AUTH: User {username} found. DB Hash: {user_row[3][:20]}...")
                # Verify password
                stored_hash = user_row[3]
                password_valid = bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))

                if not password_valid:
                    print(f"DEBUG AUTH: Password verification failed for user {username}")
                    logger.warning(f"Invalid password for user {username}")
                    return None

                print(f"DEBUG AUTH: Password verification successful for user {username}")

                # Return user data (excluding password hash)
                user_data = {
                    'id': user_row[0],
                    'username': user_row[1],
                    'email': user_row[2],
                    'fpl_id': user_row[4],
                    'subscription_plan': user_row[5]
                }

                logger.info(f"User {username} authenticated successfully")
                return user_data

        except Exception as e:
            logger.error(f"Error verifying user {username}: {e}")
            return None

    def update_fpl_id(self, user_id: int, fpl_id: str) -> bool:
        """
        Update user's FPL team ID.

        Args:
            user_id: User ID
            fpl_id: FPL team ID

        Returns:
            bool: True if updated successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE users
                    SET fpl_id = ?
                    WHERE id = ?
                """, (fpl_id, user_id))
                conn.commit()

            logger.info(f"Updated FPL ID for user {user_id}: {fpl_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating FPL ID for user {user_id}: {e}")
            return False

    def get_user_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user data by user ID.

        Args:
            user_id: User ID

        Returns:
            Optional[Dict]: User data dict if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, username, email, fpl_id, subscription_plan
                    FROM users
                    WHERE id = ?
                """, (user_id,))

                user_row = cursor.fetchone()

                if not user_row:
                    logger.warning(f"User {user_id} not found")
                    return None

                user_data = {
                    'id': user_row[0],
                    'username': user_row[1],
                    'email': user_row[2],
                    'fpl_id': user_row[4],
                    'subscription_plan': user_row[5]
                }

                return user_data

        except Exception as e:
            logger.error(f"Error getting user data for {user_id}: {e}")
            return None

    def user_exists(self, username: str) -> bool:
        """
        Check if a username already exists.

        Args:
            username: Username to check

        Returns:
            bool: True if username exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 1 FROM users WHERE username = ? LIMIT 1
                """, (username,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking if user {username} exists: {e}")
            return False

    def get_all_users(self) -> list[Dict[str, Any]]:
        """
        Get all users with their id, username, email and subscription plan.

        Returns:
            list[Dict]: List of user dictionaries containing id, username, email, subscription_plan
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, username, email, subscription_plan
                    FROM users
                    ORDER BY created_at DESC
                """)

                users = []
                for row in cursor.fetchall():
                    users.append({
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'subscription_plan': row[3]
                    })

                return users

        except Exception as e:
            logger.error(f"Error getting all users: {e}")
            return []

    def update_user_plan(self, username: str, new_plan: str) -> bool:
        """
        Update user's subscription plan.

        Args:
            username: Username to update
            new_plan: New plan ('free' or 'premium')

        Returns:
            bool: True if updated successfully
        """
        try:
            # Validate plan
            if new_plan not in ['free', 'premium']:
                logger.warning(f"Invalid plan: {new_plan}")
                return False

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE users
                    SET subscription_plan = ?
                    WHERE username = ?
                """, (new_plan, username))

                if cursor.rowcount == 0:
                    logger.warning(f"User {username} not found")
                    return False

                conn.commit()
                logger.info(f"Updated plan for user {username}: {new_plan}")
                return True

        except Exception as e:
            logger.error(f"Error updating plan for user {username}: {e}")
            return False

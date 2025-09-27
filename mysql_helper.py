# utils/mysql_helper.py
import mysql.connector
from mysql.connector import Error

# --- Database connection settings ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",        # your MySQL user
    "password": "Atharva1@",  # your MySQL password
    "database": "stock_chatbot"
}

# --- User Functions ---

def create_user(username, password):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Check if username exists
        cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
        result = cursor.fetchone()
        if result:
            return False  # username exists

        # Insert new user
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, password)
        )
        conn.commit()
        return True

    except Error as e:
        print("Error:", e)
        return False
cursor = None
conn = None
try:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    # your logic
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()



def authenticate_user(username, password):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM users WHERE username=%s AND password=%s",
            (username, password)
        )
        result = cursor.fetchone()
        if result:
            return result[0]  # Return user_id
        return None
    except mysql.connector.Error as e:
        print("DB Error:", e)
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# --- Watchlist Functions ---

def get_watchlist(user_id):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker FROM watchlist WHERE user_id=%s",
            (user_id,)
        )
        result = cursor.fetchall()
        return [row[0] for row in result]
    except mysql.connector.Error as e:
        print("DB Error:", e)
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def add_to_watchlist(user_id, ticker):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        # Check if already exists
        cursor.execute(
            "SELECT id FROM watchlist WHERE user_id=%s AND ticker=%s",
            (user_id, ticker)
        )
        if cursor.fetchone():
            return False  # Already in watchlist
        # Insert
        cursor.execute(
            "INSERT INTO watchlist (user_id, ticker) VALUES (%s, %s)",
            (user_id, ticker)
        )
        conn.commit()
        return True
    except mysql.connector.Error as e:
        print("DB Error:", e)
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def remove_from_watchlist(user_id, ticker):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM watchlist WHERE user_id=%s AND ticker=%s",
            (user_id, ticker)
        )
        conn.commit()
        return True
    except mysql.connector.Error as e:
        print("DB Error:", e)
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

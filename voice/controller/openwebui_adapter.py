"""
Адаптер для записи голосовых чатов в Open WebUI БД
"""
import sqlite3
import json
import uuid
import time
from datetime import datetime

class OpenWebUIAdapter:
    def __init__(self, db_path: str, user_id: str, auto_load_last: bool = True):
        self.db_path = db_path
        self.user_id = user_id
        self.current_chat_id = None
        self.messages = {}  # {uuid: message}
        self.message_order = []  # [uuid1, uuid2, ...]
        
        if auto_load_last:
            self._load_last_chat()
    
    def _load_last_chat(self):
        """Загрузить последний голосовой чат пользователя"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ищем последний чат с тегом voice или эмодзи 🎤 в title
        cursor.execute("""
            SELECT id, chat FROM chat 
            WHERE user_id = ? 
            AND (meta LIKE '%voice%' OR title LIKE '%🎤%')
            ORDER BY updated_at DESC 
            LIMIT 1
        """, (self.user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            chat_id, chat_json = row
            chat_data = json.loads(chat_json)
            
            self.current_chat_id = chat_id
            self.messages = chat_data.get("history", {}).get("messages", {})
            
            # Восстанавливаем order из messages array
            messages_array = chat_data.get("messages", [])
            self.message_order = [msg["id"] for msg in messages_array]
        else:
            # Нет голосовых чатов - создаем новый
            self.create_new_chat()
        
    def create_new_chat(self, title: str = "🎤 Голосовой чат"):
        """Создать новый чат"""
        self.current_chat_id = str(uuid.uuid4())
        self.messages = {}
        self.message_order = []
        
        now = int(time.time())
        
        chat_data = {
            "id": self.current_chat_id,
            "title": title,
            "models": ["qwen3:14b"],
            "params": {},
            "history": {
                "messages": {},
                "currentId": None
            },
            "messages": [],
            "tags": [],
            "timestamp": now * 1000,
            "files": []
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat (id, user_id, title, share_id, archived, created_at, updated_at, chat, pinned, meta, folder_id)
            VALUES (?, ?, ?, NULL, 0, ?, ?, ?, 0, '{"tags": ["voice"]}', NULL)
        """, (
            self.current_chat_id,
            self.user_id,
            title,
            now,
            now,
            json.dumps(chat_data)
        ))
        conn.commit()
        conn.close()
        
        return self.current_chat_id
    
    def add_message(self, role: str, content: str):
        """Добавить сообщение в текущий чат"""
        if not self.current_chat_id:
            self.create_new_chat()
        
        msg_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        parent_id = self.message_order[-1] if self.message_order else None
        
        message = {
            "id": msg_id,
            "parentId": parent_id,
            "childrenIds": [],
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "models": ["qwen3:14b"] if role == "user" else None
        }
        
        if role == "assistant":
            message.update({
                "model": "qwen3:14b",
                "modelName": "qwen3:14b",
                "modelIdx": 0,
                "done": True
            })
        
        # Обновляем parent's childrenIds
        if parent_id and parent_id in self.messages:
            self.messages[parent_id]["childrenIds"].append(msg_id)
        
        self.messages[msg_id] = message
        self.message_order.append(msg_id)
        
        # Сохраняем в БД
        self._save_to_db()
        
        return msg_id
    
    def _save_to_db(self):
        """Сохранить текущее состояние в БД"""
        now = int(time.time())
        
        # Формируем messages array для отображения
        messages_array = [self.messages[msg_id] for msg_id in self.message_order]
        
        chat_data = {
            "id": self.current_chat_id,
            "title": "🎤 Голосовой чат",
            "models": ["qwen3:14b"],
            "params": {},
            "history": {
                "messages": self.messages,
                "currentId": self.message_order[-1] if self.message_order else None
            },
            "messages": messages_array,
            "tags": [],
            "timestamp": now * 1000,  # default
            "files": []
        }
        
        # Генерируем title из первого user сообщения
        first_user_msg = next((m for m in messages_array if m["role"] == "user"), None)
        if first_user_msg:
            title_text = first_user_msg["content"][:50]
            if len(first_user_msg["content"]) > 50:
                title_text += "..."
            chat_data["title"] = f"🎤 {title_text}"
            # Берем timestamp из первого сообщения
            chat_data["timestamp"] = int(first_user_msg["timestamp"] * 1000)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat 
            SET updated_at = ?, 
                chat = ?,
                title = ?
            WHERE id = ?
        """, (
            now,
            json.dumps(chat_data),
            chat_data["title"],
            self.current_chat_id
        ))
        conn.commit()
        conn.close()
    
    def get_current_chat_id(self):
        """Получить ID текущего чата"""
        return self.current_chat_id
    
    def get_history(self):
        """Получить историю сообщений текущего чата в формате для LLM"""
        return [{"role": msg["role"], "content": msg["content"]} 
                for msg in [self.messages[msg_id] for msg_id in self.message_order]]
    
    def get_message_count(self):
        """Получить количество сообщений в текущем чате"""
        return len(self.message_order)

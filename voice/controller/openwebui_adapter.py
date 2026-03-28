"""
Адаптер для записи голосовых чатов в Open WebUI БД.
Если БД недоступна или таблица chat отсутствует — работает в памяти без персистентности.
"""
import os
import sqlite3
import json
import uuid
import time
import logging

log = logging.getLogger("controller.webui")

_LLM_MODEL = os.environ.get("LLM_MODEL", "unknown")


class OpenWebUIAdapter:
    def __init__(self, db_path: str, user_id: str, auto_load_last: bool = True):
        self.db_path = db_path
        self.user_id = user_id
        self.current_chat_id = None
        self.messages = {}        # {uuid: message}
        self.message_order = []   # [uuid1, uuid2, ...]
        self._db_available = False

        self._check_db()

        if auto_load_last and self._db_available:
            try:
                self._load_last_chat()
            except Exception as e:
                log.warning("Could not load last chat from DB: %s — starting fresh in-memory session", e)
                self._init_memory_session()
        else:
            self._init_memory_session()

    def _check_db(self):
        """Проверяет, доступна ли БД и существует ли таблица chat."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat'")
            if cursor.fetchone():
                self._db_available = True
                log.info("Open WebUI DB available: %s", self.db_path)
            else:
                log.warning(
                    "Open WebUI DB found at %s but table 'chat' is missing — "
                    "running without history persistence. "
                    "Make sure the open-webui volume is shared with this container.",
                    self.db_path,
                )
            conn.close()
        except Exception as e:
            log.warning(
                "Open WebUI DB not accessible (%s: %s) — running without history persistence.",
                self.db_path, e,
            )

    def _init_memory_session(self):
        """Инициализирует пустую сессию в памяти (без записи в БД)."""
        self.current_chat_id = str(uuid.uuid4())
        self.messages = {}
        self.message_order = []

    def _load_last_chat(self):
        """Загрузить последний голосовой чат пользователя из БД."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
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
            messages_array = chat_data.get("messages", [])
            self.message_order = [msg["id"] for msg in messages_array]
        else:
            self.create_new_chat()

    def create_new_chat(self, title: str = "🎤 Голосовой чат"):
        """Создать новый чат (в БД если доступна, иначе в памяти)."""
        self.current_chat_id = str(uuid.uuid4())
        self.messages = {}
        self.message_order = []

        if not self._db_available:
            return self.current_chat_id

        now = int(time.time())
        chat_data = {
            "id": self.current_chat_id,
            "title": title,
            "models": [_LLM_MODEL],
            "params": {},
            "history": {"messages": {}, "currentId": None},
            "messages": [],
            "tags": [],
            "timestamp": now * 1000,
            "files": [],
        }
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat (id, user_id, title, share_id, archived, created_at, updated_at, chat, pinned, meta, folder_id)
                VALUES (?, ?, ?, NULL, 0, ?, ?, ?, 0, '{"tags": ["voice"]}', NULL)
            """, (self.current_chat_id, self.user_id, title, now, now, json.dumps(chat_data)))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("Could not create chat in DB: %s", e)

        return self.current_chat_id

    def add_message(self, role: str, content: str):
        """Добавить сообщение в текущий чат."""
        if not self.current_chat_id:
            self._init_memory_session()

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
            "models": [_LLM_MODEL] if role == "user" else None,
        }
        if role == "assistant":
            message.update({
                "model": _LLM_MODEL,
                "modelName": _LLM_MODEL,
                "modelIdx": 0,
                "done": True,
            })

        if parent_id and parent_id in self.messages:
            self.messages[parent_id]["childrenIds"].append(msg_id)

        self.messages[msg_id] = message
        self.message_order.append(msg_id)

        if self._db_available:
            try:
                self._save_to_db()
            except Exception as e:
                log.warning("Could not save message to DB: %s", e)

        return msg_id

    def _save_to_db(self):
        """Сохранить текущее состояние в БД."""
        now = int(time.time())
        messages_array = [self.messages[mid] for mid in self.message_order]

        chat_data = {
            "id": self.current_chat_id,
            "title": "🎤 Голосовой чат",
            "models": [_LLM_MODEL],
            "params": {},
            "history": {
                "messages": self.messages,
                "currentId": self.message_order[-1] if self.message_order else None,
            },
            "messages": messages_array,
            "tags": [],
            "timestamp": now * 1000,
            "files": [],
        }

        first_user_msg = next((m for m in messages_array if m["role"] == "user"), None)
        if first_user_msg:
            title_text = first_user_msg["content"][:50]
            if len(first_user_msg["content"]) > 50:
                title_text += "..."
            chat_data["title"] = f"🎤 {title_text}"
            chat_data["timestamp"] = int(first_user_msg["timestamp"] * 1000)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat SET updated_at = ?, chat = ?, title = ? WHERE id = ?
        """, (now, json.dumps(chat_data), chat_data["title"], self.current_chat_id))
        conn.commit()
        conn.close()

    def get_current_chat_id(self):
        return self.current_chat_id

    def get_history(self):
        """История сообщений в формате для LLM."""
        return [
            {"role": self.messages[mid]["role"], "content": self.messages[mid]["content"]}
            for mid in self.message_order
        ]

    def get_message_count(self):
        return len(self.message_order)

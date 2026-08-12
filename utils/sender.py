"""This module provides a fault-tolerant email notification service."""

import os
import smtplib
from email.message import EmailMessage
from typing import Optional

class EmailSender:
    """
    A notification service that sends email alerts via SMTP.
    Non-sensitive parameters are configurable via a dictionary (e.g., from TOML),
    while credentials are strictly read from environment variables (.env).
    """
    def __init__(self, config_dict: Optional[dict] = None):
        # 1. Set default fallback parameters
        self.receiver_email = "admin@example.com"
        self.smtp_server = "smtp.gmail.com"
        self.port = 587
        self.subject_prefix = "YOLO Trainer Alert"
        
        # 2. Override defaults with values from the configuration file, if provided
        if config_dict:
            self.receiver_email = config_dict.get("receiver_email", self.receiver_email)
            self.smtp_server = config_dict.get("smtp_server", self.smtp_server)
            self.port = config_dict.get("port", self.port)
            self.subject_prefix = config_dict.get("subject_prefix", self.subject_prefix)
            
        # 3. Strictly fetch sensitive credentials from environment variables (.env)
        self.sender_email = os.environ.get("SENDER_EMAIL")
        self.sender_password = os.environ.get("SENDER_PASSWORD")

    def send_message(self, message: str) -> None:
        """
        Constructs and sends an email containing the log message.
        Raises exceptions upon failure, which are safely handled by the Logger's decorator.
        """
        if not self.sender_email or not self.sender_password:
            raise ValueError("Email credentials (SENDER_EMAIL, SENDER_PASSWORD) are missing in environment variables.")

        try:
            msg = EmailMessage()
            msg.set_content(message)
            msg["Subject"] = self.subject_prefix
            msg["From"] = self.sender_email
            msg["To"] = self.receiver_email

            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                
        except smtplib.SMTPAuthenticationError:
            raise PermissionError("Authentication failed. Please check your App Password.")
            
        except Exception as e:
            raise RuntimeError(f"An unexpected SMTP error occurred: {e}")
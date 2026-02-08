import datetime
import random
from dataclasses import dataclass


@dataclass
class Response:
    text: str
    confidence: float


class SimpleAI:
    """A tiny rule-based assistant for demo purposes."""

    def __init__(self) -> None:
        self.fallbacks = [
            "Ainda estou aprendendo. Pode reformular?",
            "Não tenho certeza. Pode dar mais detalhes?",
            "Interessante! Conta mais para eu entender melhor.",
        ]

    def respond(self, message: str) -> Response:
        normalized = message.strip().lower()
        if not normalized:
            return Response("Pode escrever sua pergunta?", 0.2)

        if any(word in normalized for word in ["oi", "olá", "ola", "hello"]):
            return Response("Olá! Como posso ajudar hoje?", 0.9)

        if "hora" in normalized or "tempo" in normalized:
            now = datetime.datetime.now().strftime("%H:%M")
            return Response(f"Agora são {now}.", 0.8)

        if "seu nome" in normalized or "quem é você" in normalized:
            return Response("Sou uma IA simples criada para demonstração.", 0.7)

        if "obrig" in normalized:
            return Response("De nada!", 0.9)

        fallback = random.choice(self.fallbacks)
        return Response(fallback, 0.3)

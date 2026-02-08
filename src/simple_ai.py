import datetime
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Response:
    text: str
    confidence: float
    intent: str


class SimpleAI:
    """A tiny Naive Bayes classifier for demo purposes."""

    def __init__(self) -> None:
        self.training_data = {
            "greeting": [
                "oi",
                "olá",
                "ola",
                "e aí",
                "bom dia",
                "boa tarde",
                "boa noite",
                "hello",
                "hi",
            ],
            "time": [
                "que horas são",
                "qual é a hora",
                "me diz a hora",
                "que horas",
                "hora agora",
            ],
            "identity": [
                "quem é você",
                "qual seu nome",
                "o que você é",
                "você é uma ia",
                "me fale sobre você",
            ],
            "gratitude": [
                "obrigado",
                "obrigada",
                "valeu",
                "agradeço",
            ],
            "goodbye": [
                "tchau",
                "até mais",
                "até logo",
                "falou",
            ],
        }
        self.responses = {
            "greeting": "Olá! Como posso ajudar hoje?",
            "time": "Agora são {time}.",
            "identity": "Sou uma IA de demonstração treinada com exemplos simples.",
            "gratitude": "De nada!",
            "goodbye": "Até mais!",
            "fallback": "Ainda estou aprendendo. Pode dar mais detalhes?",
        }
        self._fit_model()

    def _fit_model(self) -> None:
        self.vocabulary: set[str] = set()
        self.class_word_counts: dict[str, Counter[str]] = {}
        self.class_totals: dict[str, int] = {}
        self.class_priors: dict[str, float] = {}

        total_examples = sum(len(samples) for samples in self.training_data.values())
        for intent, samples in self.training_data.items():
            word_counter = Counter()
            for sample in samples:
                tokens = self._tokenize(sample)
                word_counter.update(tokens)
                self.vocabulary.update(tokens)
            self.class_word_counts[intent] = word_counter
            self.class_totals[intent] = sum(word_counter.values())
            self.class_priors[intent] = len(samples) / total_examples

    def _tokenize(self, text: str) -> list[str]:
        normalized = re.sub(r"[^\w\sáàãâéêíóôõúç]", " ", text.lower())
        return [token for token in normalized.split() if token]

    def _score_intent(self, tokens: Iterable[str], intent: str) -> float:
        log_prob = math.log(self.class_priors[intent])
        vocab_size = len(self.vocabulary)
        word_counts = self.class_word_counts[intent]
        total_words = self.class_totals[intent]
        for token in tokens:
            count = word_counts.get(token, 0) + 1
            log_prob += math.log(count / (total_words + vocab_size))
        return log_prob

    def respond(self, message: str) -> Response:
        normalized = message.strip()
        if not normalized:
            return Response("Pode escrever sua pergunta?", 0.1, "fallback")

        tokens = self._tokenize(normalized)
        if not tokens:
            return Response("Pode escrever sua pergunta?", 0.1, "fallback")

        scores = {
            intent: self._score_intent(tokens, intent)
            for intent in self.training_data.keys()
        }
        best_intent = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        confidence = 0.5
        if len(sorted_scores) > 1:
            confidence = 1 / (1 + math.exp(sorted_scores[1] - sorted_scores[0]))

        if confidence < 0.45:
            fallback = self.responses["fallback"]
            return Response(fallback, confidence, "fallback")

        if best_intent == "time":
            now = datetime.datetime.now().strftime("%H:%M")
            return Response(self.responses["time"].format(time=now), confidence, best_intent)

        if best_intent == "goodbye":
            farewell = self.responses["goodbye"]
            if random.random() < 0.4:
                farewell = "Até logo!"
            return Response(farewell, confidence, best_intent)

        return Response(self.responses[best_intent], confidence, best_intent)

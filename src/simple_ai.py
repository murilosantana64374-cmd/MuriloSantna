import datetime
import math
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Response:
    text: str
    confidence: float
    intent: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Example:
    text: str
    intent: str


class IntentModel:
    def __init__(self, examples: Iterable[Example]) -> None:
        self.examples = list(examples)
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self.intent_centroids: dict[str, list[float]] = {}
        self._fit()

    def _fit(self) -> None:
        documents = [self._tokenize(example.text) for example in self.examples]
        vocab: dict[str, int] = {}
        for tokens in documents:
            for token in set(tokens):
                vocab.setdefault(token, len(vocab))
        self.vocabulary = vocab

        doc_count = len(documents)
        doc_freq = [0] * len(vocab)
        for tokens in documents:
            seen = set(tokens)
            for token in seen:
                doc_freq[vocab[token]] += 1
        self.idf = [math.log((1 + doc_count) / (1 + df)) + 1 for df in doc_freq]

        intent_vectors: dict[str, list[list[float]]] = {}
        for example, tokens in zip(self.examples, documents, strict=False):
            vector = self._vectorize(tokens)
            intent_vectors.setdefault(example.intent, []).append(vector)

        self.intent_centroids = {
            intent: self._average(vectors) for intent, vectors in intent_vectors.items()
        }

    def _tokenize(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFD", text.lower())
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        return [token for token in normalized.split() if token]

    def _vectorize(self, tokens: list[str]) -> list[float]:
        if not self.vocabulary:
            return []
        vector = [0.0] * len(self.vocabulary)
        counts: dict[int, int] = {}
        for token in tokens:
            if token in self.vocabulary:
                index = self.vocabulary[token]
                counts[index] = counts.get(index, 0) + 1
        if not counts:
            return vector
        max_count = max(counts.values())
        for index, count in counts.items():
            tf = count / max_count
            vector[index] = tf * self.idf[index]
        return vector

    def _average(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        length = len(vectors[0])
        totals = [0.0] * length
        for vector in vectors:
            for i, value in enumerate(vector):
                totals[i] += value
        return [value / len(vectors) for value in totals]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        numerator = sum(x * y for x, y in zip(a, b, strict=False))
        denom_a = math.sqrt(sum(x * x for x in a))
        denom_b = math.sqrt(sum(y * y for y in b))
        if denom_a == 0 or denom_b == 0:
            return 0.0
        return numerator / (denom_a * denom_b)

    def predict(self, text: str) -> tuple[str, float]:
        tokens = self._tokenize(text)
        vector = self._vectorize(tokens)
        best_intent = "fallback"
        best_score = 0.0
        for intent, centroid in self.intent_centroids.items():
            score = self._cosine_similarity(vector, centroid)
            if score > best_score:
                best_score = score
                best_intent = intent
        return best_intent, best_score


class SimpleAI:
    """A tiny assistant with intent detection and a bit of memory."""

    def __init__(self) -> None:
        self.examples = [
            Example("oi", "greeting"),
            Example("olá", "greeting"),
            Example("bom dia", "greeting"),
            Example("boa tarde", "greeting"),
            Example("boa noite", "greeting"),
            Example("hello", "greeting"),
            Example("que horas são", "time"),
            Example("qual é a hora", "time"),
            Example("me diz a hora", "time"),
            Example("hora agora", "time"),
            Example("quem é você", "identity"),
            Example("qual seu nome", "identity"),
            Example("o que você é", "identity"),
            Example("você é uma ia", "identity"),
            Example("me fale sobre você", "identity"),
            Example("obrigado", "gratitude"),
            Example("obrigada", "gratitude"),
            Example("valeu", "gratitude"),
            Example("agradeço", "gratitude"),
            Example("tchau", "goodbye"),
            Example("até mais", "goodbye"),
            Example("até logo", "goodbye"),
            Example("falou", "goodbye"),
            Example("preciso de ajuda", "help"),
            Example("o que você faz", "help"),
            Example("me ajude", "help"),
            Example("conte uma piada", "joke"),
            Example("tem uma piada", "joke"),
        ]
        self.model = IntentModel(self.examples)
        self.user_name: str | None = None
        self.last_intent: str | None = None
        self.intent_keywords = {
            "time": {"hora", "horas", "tempo"},
            "gratitude": {"obrigado", "obrigada", "valeu"},
            "goodbye": {"tchau", "falou", "adeus"},
            "help": {"ajuda", "ajude", "comandos"},
            "joke": {"piada", "engraçado"},
        }
        self.responses = {
            "greeting": "Olá{suffix}! Como posso ajudar hoje?",
            "time": "Agora são {time}.",
            "identity": "Sou uma IA de demonstração treinada com exemplos simples.",
            "gratitude": "De nada!",
            "goodbye": "Até mais!",
            "help": "Posso responder cumprimentos, hora, identidade, piadas e guardar seu nome.",
            "joke": "Por que o computador foi ao médico? Porque estava com um vírus.",
            "fallback": "Ainda estou aprendendo. Pode dar mais detalhes?",
        }

    def add_example(self, text: str, intent: str) -> None:
        self.examples.append(Example(text, intent))
        self.model = IntentModel(self.examples)

    def _normalize_text(self, message: str) -> str:
        normalized = unicodedata.normalize("NFD", message.lower())
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return normalized

    def _extract_name(self, message: str) -> str | None:
        normalized = self._normalize_text(message)
        match = re.search(
            r"\b(meu nome e|me chamo|eu sou)\s+(?P<name>[a-z]+)",
            normalized,
        )
        if match:
            name = match.group("name").strip().capitalize()
            return name
        return None

    def _boost_intent(self, message: str, base_intent: str, base_score: float) -> float:
        tokens = set(self.model._tokenize(message))
        bonus = 0.0
        for intent, keywords in self.intent_keywords.items():
            if intent == base_intent and keywords.intersection(tokens):
                bonus += 0.15
        return min(base_score + bonus, 1.0)

    def respond(self, message: str) -> Response:
        normalized = message.strip()
        if not normalized:
            return Response("Pode escrever sua pergunta?", 0.1, "fallback")

        name = self._extract_name(normalized)
        if name:
            self.user_name = name
            return Response(
                f"Prazer, {name}! Como posso ajudar?",
                0.9,
                "name",
                {"name": name},
            )

        if self.last_intent == "identity" and "e voce" in self._normalize_text(
            normalized
        ):
            return Response(self.responses["identity"], 0.7, "identity")

        best_intent, score = self.model.predict(normalized)
        confidence = self._boost_intent(normalized, best_intent, score)

        if confidence < 0.25:
            return Response(self.responses["fallback"], confidence, "fallback")

        if best_intent == "time":
            now = datetime.datetime.now().strftime("%H:%M")
            response_text = self.responses["time"].format(time=now)
        elif best_intent == "greeting":
            suffix = f", {self.user_name}" if self.user_name else ""
            response_text = self.responses["greeting"].format(suffix=suffix)
        else:
            response_text = self.responses.get(best_intent, self.responses["fallback"])

        if best_intent == "goodbye" and random.random() < 0.4:
            response_text = "Até logo!"

        self.last_intent = best_intent
        return Response(response_text, confidence, best_intent)

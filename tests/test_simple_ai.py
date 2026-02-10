import unittest

from src.simple_ai import SimpleAI


class SimpleAITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ai = SimpleAI()

    def test_greeting_intent(self) -> None:
        response = self.ai.respond("Olá")
        self.assertEqual(response.intent, "greeting")
        self.assertGreaterEqual(response.confidence, 0.25)

    def test_name_memory(self) -> None:
        response = self.ai.respond("Meu nome é Ana")
        self.assertEqual(response.intent, "name")
        follow_up = self.ai.respond("oi")
        self.assertIn("Ana", follow_up.text)

    def test_help_intent(self) -> None:
        response = self.ai.respond("preciso de ajuda")
        self.assertEqual(response.intent, "help")

    def test_time_intent(self) -> None:
        response = self.ai.respond("que horas são")
        self.assertEqual(response.intent, "time")


if __name__ == "__main__":
    unittest.main()

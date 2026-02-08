from simple_ai import SimpleAI


def run() -> None:
    assistant = SimpleAI()
    print("IA pronta! Digite 'sair' para encerrar.")
    while True:
        message = input("> ")
        if message.strip().lower() in {"sair", "exit", "quit"}:
            print("Até mais!")
            break
        response = assistant.respond(message)
        print(f"{response.text} (confiança: {response.confidence:.1f})")


if __name__ == "__main__":
    run()

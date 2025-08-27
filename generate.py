import torch
from main import TinyFlashcardLLM, FlashcardDataset

def generate_flashcards(prompt, max_new_tokens=50):
    dataset = FlashcardDataset("data/flashcards.json")
    model = TinyFlashcardLLM(dataset.vocab_size)
    model.load_state_dict(torch.load("tinyflashcards.pth"))
    model.eval()

    x = torch.tensor([dataset.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    return dataset.decode(x[0].tolist())

if __name__ == "__main__":
    context = "The mitochondria is an organelle that produces ATP for energy."
    print(generate_flashcards("Context: " + context + " Q:"))

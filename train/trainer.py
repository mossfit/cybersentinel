def train(model, loader, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        for img_high, img_low, labels in loader:
            preds = model(img_high, img_low)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

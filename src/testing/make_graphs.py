import matplotlib.pyplot as plt
import pickle

# Load the training history
with open('src/model_stats.pkl', 'rb') as f:
    history = pickle.load(f)

print(history)

# avg_loss, avg_acc, avg_prp
# avg_acc, avg_prp

# Extracting the average loss, accuracy, and perplexity from the history
avg_loss = [epoch_data[0] for epoch_data in history[::2]]  # Every second element starting from 0
avg_acc = [epoch_data[1] for epoch_data in history[::2]]   # Every second element starting from 0
avg_prp = [epoch_data[2] for epoch_data in history[::2]]   # Every second element starting from 0avg_loss = [epoch_data[0] for epoch_data in history[::2]]  # Every second element starting from 0
test_prp = [epoch_data[0] for epoch_data in history[1::2]]   # Every second element starting from 0
test_acc = [epoch_data[1] for epoch_data in history[1::2]]   # Every second element starting from 0

# Epochs for the x-axis
epochs = range(1, len(avg_loss) + 1)

# Plotting the average loss and perplexity
plt.figure(figsize=(10, 7))
plt.plot(epochs, avg_loss, label='Average Loss', marker='o')
plt.plot(epochs, avg_prp, label='Average Perplexity', marker='o')
plt.plot(epochs, test_prp, label='Test Perplexity', marker='x')

plt.title('Loss and Perplexity Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss/Perplexity')
plt.legend()
plt.grid(True)
# Ensure x-axis ticks are integers
plt.xticks(epochs)

# Plotting the accuracy
plt.figure(figsize=(10, 7))
plt.plot(epochs, avg_acc, label='Average Accuracy', marker='o')
plt.plot(epochs, test_acc, label='Test Accuracy', marker='x')

plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# Ensure x-axis ticks are integers
plt.xticks(epochs)

plt.show()

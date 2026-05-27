from sklearn.metrics import accuracy_score, precision_score, recall_score

# Actual answers
y_true = [1, 1, 0, 1, 0]

# Chatbot predictions
y_pred = [1, 1, 0, 0, 0]

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Precision
precision = precision_score(y_true, y_pred)

# Recall
recall = recall_score(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
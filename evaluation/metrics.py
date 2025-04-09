from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    return accuracy, precision, recall, f1

from preprocessing.dataset_build import build_dataset
from training.train import train_model
from evaluation.evaluate import evaluate_model

def main():
    build_dataset('./dataset/raw_pe_files/', './dataset/images/')
    train_model()
    evaluate_model()

if __name__ == "__main__":
    main()

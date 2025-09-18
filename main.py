from src.data_preprocessing import load_and_preprocess
from src.model_training import train_models
from src.evaluation import evaluate
from src.visualization import plot_confusion_matrix, plot_roc, plot_feature_importance

def main():
    # Load + preprocess
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

    # Train
    models = train_models(X_train, y_train)

    # Evaluate
    results = evaluate(models, X_test, y_test)

    # Show results
    for name, res in results.items():
        print(f"\n=== {name} ===")
        print(f"Accuracy: {res['accuracy']:.2f}")
        print("Confusion Matrix:\n", res['confusion_matrix'])
        print("Classification Report:\n", res['report'])

        # Plots
        plot_confusion_matrix(res['confusion_matrix'], name)
        plot_roc(models[name], X_test, y_test, name)

        if name == "Random Forest":
            plot_feature_importance(models[name], feature_names)

if __name__ == "__main__":
    main()

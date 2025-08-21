from preprocessing import load_data, get_target, scale_features
from predictor import split_data, train_model, eval_model

def main():
    filepath = 'placement-dataset.csv'
    data = load_data(filepath)

    acc_cgpa, prec_cgpa, rec_cgpa = pipeline(data, ['cgpa'])
    acc_iq, prec_iq, rec_iq = pipeline(data, ['iq'])
    acc_both, prec_both, rec_both = pipeline(data, ['cgpa', 'iq'])

    print()
    print_metrics("CGPA Only", acc_cgpa, prec_cgpa, rec_cgpa)
    print_metrics("IQ Only", acc_iq, prec_iq, rec_iq)
    print_metrics("CGPA + IQ", acc_both, prec_both, rec_both)

    if acc_both > max(acc_cgpa, acc_iq):
        print("CGPA and IQ together are better predictors of placement.")
    elif acc_cgpa > acc_iq:
        print("CGPA is the stronger individual predictor.")
    else:
        print("IQ is the stronger individual predictor.")

def pipeline(data, feature_names):
    """trains the model and returns the accuracy, precision and recall features"""
    X = scale_features(data, feature_names)
    y = get_target(data)

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    acc, prec, rec = eval_model(model, X_test, y_test)

    return acc, prec, rec

def print_metrics(name, acc, prec, rec):
    print(f" {name}")
    print(f"  Accuracy : {acc:.2f}")
    print(f"  Precision: {prec:.2f}")
    print(f"  Recall   : {rec:.2f}")
    print()

if __name__ == '__main__':
    main()

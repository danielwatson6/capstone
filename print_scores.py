import os
import sys


if __name__ == "__main__":
    path = sys.argv[1]
    with open(os.path.join(path, "estimates.tsv")) as f:
        current_model = None
        current_scores = []
        scores = []
        for i, line in enumerate(f):
            if i == 0:
                continue
            try:
                model, _, _, score = line.strip().split("\t")
            except ValueError:
                continue

            if model != current_model:
                if current_model is not None:
                    print(current_model)
                    for s in current_scores:
                        print("  {:.4f}".format(s))
                    try:
                        if current_model.startswith("ub"):
                            best_score = min(current_scores)
                        else:
                            best_score = max(current_scores)
                    except ValueError:
                        best_score = float("nan")
                    scores.append((current_model, best_score))

                current_model = model
                current_scores = []

            if score != "nan":
                current_scores.append(float(score))

        print(current_model)
        for s in current_scores:
            print("  {:.4f}".format(s))
        try:
            if model.startswith("ub"):
                best_score = min(current_scores)
            else:
                best_score = max(current_scores)
        except ValueError:
            best_score = float("nan")
        scores.append((current_model, best_score))

    print("\nBest scores:")
    for model, score in scores:
        print("  {}\t{:.4f}".format(model, score))

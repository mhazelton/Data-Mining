# Max Hazelton, CISC-4631-R01
# Bayes Classifier Lab
# Reads dataset Last, column = class label.
# Here are some example queries:
#   age=<=30,income=high,student=no,credit_rating=fair
#   age=>40,income=medium,student=no,credit_rating=fair
#   Outlook=overcast,Temperature=hot,Humidity=high,Windy=false
#   Outlook=rain,Temperature=cool,Humidity=normal,Windy=true 

import math
from collections import defaultdict, Counter
from pathlib import Path

def read_table(path):
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    header = lines[0]
    if "," in header:
        headers = [h.strip() for h in header.split(",")]
        split = lambda s: [x.strip() for x in s.split(",")]
    else:
        headers = header.split()
        split = lambda s: s.split()
    data = []
    for line in lines[1:]:
        parts = split(line)
        data.append({headers[i]: parts[i] for i in range(len(headers))})
    return headers, data

class NaiveBayes:
    def __init__(self):
        self.class_counts = Counter()
        self.feature_counts = defaultdict(lambda: defaultdict(Counter))
        self.feature_values = defaultdict(set)
        self.features = []
        self.label = ""
        self.n = 0

    def fit(self, headers, rows):
        self.label = headers[-1]
        self.features = headers[:-1]
        self.n = len(rows)
        for r in rows:
            c = r[self.label]
            self.class_counts[c] += 1
            for f in self.features:
                v = r[f]
                self.feature_counts[f][c][v] += 1
                self.feature_values[f].add(v)

    def predict(self, query):
        probs = {}
        for c in self.class_counts:
            logp = math.log(self.class_counts[c] / self.n)
            for f, v in query.items():
                count = self.feature_counts[f][c][v]
                V = len(self.feature_values[f]) or 1
                logp += math.log((count + 1) / (self.class_counts[c] + V))
            probs[c] = math.exp(logp)
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total
        return probs

def parse_query(s):
    q = {}
    for tok in s.split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            q[k.strip()] = v.strip()
    return q

def choose_file():
    print("Choose dataset:")
    print("  1) buys_computer.txt")
    print("  2) play_tennis.txt")
    ch = input("Enter 1 or 2 (default 1): ").strip() or "1"
    name = "buys_computer.txt" if ch == "1" else "play_tennis.txt"
    return Path(name)

def main():
    path = choose_file()
    headers, data = read_table(path)
    nb = NaiveBayes()
    nb.fit(headers, data)

    print(f"\nLoaded {len(data)} rows from {path}")
    print("Columns:", headers)
    example = ",".join(f"{f}={data[0][f]}" for f in headers[:-1])
    print("Example:", example)
    s = input("Enter query (comma-separated key=value): ").strip()
    if not s:
        s = example
    q = parse_query(s)

    probs = nb.predict(q)
    print("\nPosterior probabilities:")
    for c, p in probs.items():
        print(f"  {c}: {p:.5f}")
    pred = max(probs, key=probs.get)
    print("Predicted class:", pred)

if __name__ == "__main__":
    main()

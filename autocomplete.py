import os
class AutocompleteEngine:
    def __init__(self, wordlist_path="data/az_wordlist.txt"):
        self.words = []
        self._load(wordlist_path)

    def _load(self, path):
        if not os.path.exists(path):
            print(f"Warning: wordlist not found at {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    self.words.append(word.upper())

    def suggest(self, prefix, max_results=5):
        if not prefix:
            return []
        prefix = prefix.upper()
        matches = [w for w in self.words if w.startswith(prefix)]
        return matches[:max_results]
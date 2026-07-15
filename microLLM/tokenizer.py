from collections import Counter


class BaseTokenizer:
    name = "base"

    def _set_vocab(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.vocab_size = len(self.vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}

    def to_config(self):
        return {
            "type": self.name,
            "vocab": self.vocab,
        }

    def encode(self, s: str):
        raise NotImplementedError

    def decode(self, nums: list[int]):
        return "".join(self.itos[i] for i in nums)


class Tokenizer(BaseTokenizer):
    name = "char"

    def __init__(self, text: str, verbose: bool = True):
        self.chars = sorted(set(text))
        self._build_maps(verbose=verbose)

    @classmethod
    def from_chars(cls, chars: list[str], verbose: bool = True):
        tokenizer = cls.__new__(cls)
        tokenizer.chars = list(chars)
        tokenizer._build_maps(verbose=verbose)
        return tokenizer

    @classmethod
    def from_config(cls, config: dict, verbose: bool = True):
        return cls.from_chars(config.get("chars", config["vocab"]), verbose=verbose)

    def _build_maps(self, verbose: bool = True):
        self._set_vocab(list(self.chars))
        if verbose:
            print("\ntokenizer: char")
            print("vocab_size:", self.vocab_size)
            print("chars:", "".join(self.chars))

    def to_config(self):
        config = super().to_config()
        config["chars"] = self.chars
        return config

    def encode(self, s: str):
        return [self.stoi[c] for c in s]


class BPETokenizer(BaseTokenizer):
    name = "bpe"

    def __init__(
        self,
        text: str,
        num_merges: int = 200,
        min_frequency: int = 2,
        vocab_text: str | None = None,
        verbose: bool = True,
    ):
        if num_merges < 0:
            raise ValueError("num_merges must be >= 0.")
        if min_frequency < 2:
            raise ValueError("min_frequency should be at least 2.")

        self.initial_chars = sorted(set(vocab_text if vocab_text is not None else text))
        self.merges: list[tuple[str, str]] = []
        tokens = tuple(text)

        if verbose:
            print("\ntokenizer: bpe")
            print("training BPE tokenizer...")
            print("raw chars:", len(tokens))
            print("initial chars:", len(self.initial_chars))
            print("planned merges:", num_merges, flush=True)

        for merge_idx in range(num_merges):
            stats = self._get_pair_stats(tokens)
            if not stats:
                break

            best_pair, best_count = stats.most_common(1)[0]
            if best_count < min_frequency:
                break

            self.merges.append(best_pair)
            tokens = self._merge_pair(tokens, best_pair)
            if verbose and (
                merge_idx == 0
                or (merge_idx + 1) % 50 == 0
                or merge_idx + 1 == num_merges
            ):
                merged_token = "".join(best_pair)
                print(
                    f"BPE merge {merge_idx + 1}/{num_merges}: "
                    f"freq={best_count}, token_len={len(tokens)}, merged={merged_token!r}",
                    flush=True,
                )

        merged_tokens = ["".join(pair) for pair in self.merges]
        self.vocab = sorted(set(self.initial_chars) | set(merged_tokens) | set(tokens))
        self._build_maps(verbose=verbose)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._cached_text = text
        self._cached_ids = [self.stoi[token] for token in tokens]

    @classmethod
    def from_config(cls, config: dict, verbose: bool = True):
        tokenizer = cls.__new__(cls)
        tokenizer.initial_chars = list(config["initial_chars"])
        tokenizer.merges = [tuple(pair) for pair in config["merges"]]
        if "vocab" in config:
            tokenizer.vocab = list(config["vocab"])
        else:
            merged_tokens = ["".join(pair) for pair in tokenizer.merges]
            tokenizer.vocab = sorted(set(tokenizer.initial_chars) | set(merged_tokens))
        tokenizer._build_maps(verbose=verbose)
        tokenizer.merge_ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}
        return tokenizer

    @staticmethod
    def _get_pair_stats(tokens: tuple[str, ...]):
        return Counter(zip(tokens, tokens[1:]))

    @staticmethod
    def _merge_pair(tokens: tuple[str, ...], pair: tuple[str, str]):
        merged_token = "".join(pair)
        out = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                out.append(merged_token)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return tuple(out)

    def _build_maps(self, verbose: bool = True):
        self._set_vocab(self.vocab)

        if verbose:
            preview = " / ".join(self.vocab[:40])
            print("initial chars:", len(self.initial_chars))
            print("merge count:", len(self.merges))
            print("vocab_size:", self.vocab_size)
            print("vocab preview:", preview, flush=True)

    def to_config(self):
        config = super().to_config()
        config.update(
            {
                "initial_chars": self.initial_chars,
                "merges": [list(pair) for pair in self.merges],
            }
        )
        return config

    def encode(self, s: str):
        if getattr(self, "_cached_text", None) == s:
            return list(self._cached_ids)
        return self._encode_chunks(s, show_progress=False)

    def encode_with_progress(self, s: str, chunk_size: int = 4096):
        if getattr(self, "_cached_text", None) == s:
            print("reusing cached BPE token ids.", flush=True)
            return list(self._cached_ids)
        return self._encode_chunks(s, chunk_size=chunk_size, show_progress=True)

    def _encode_chunks(self, s: str, chunk_size: int = 4096, show_progress: bool = False):
        ids = []
        total = len(s)
        for start in range(0, total, chunk_size):
            piece = s[start:start + chunk_size]
            ids.extend(self._encode_piece(piece))
            if show_progress and (
                start == 0
                or start + chunk_size >= total
                or (start // chunk_size + 1) % 500 == 0
            ):
                done = min(start + chunk_size, total)
                print(f"BPE encode: {done}/{total} chars", flush=True)
        return ids

    def _encode_piece(self, piece: str):
        tokens = list(piece)
        if not tokens:
            return []

        while len(tokens) >= 2:
            best_rank = None
            best_pair = None
            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            tokens = list(self._merge_pair(tuple(tokens), best_pair))

        return [self.stoi[token] for token in tokens]


def build_tokenizer(
    text: str,
    tokenizer_type: str = "char",
    bpe_merges: int = 200,
    bpe_min_frequency: int = 2,
    vocab_text: str | None = None,
    verbose: bool = True,
):
    if tokenizer_type == "char":
        return Tokenizer(text, verbose=verbose)
    if tokenizer_type == "bpe":
        return BPETokenizer(
            text,
            num_merges=bpe_merges,
            min_frequency=bpe_min_frequency,
            vocab_text=vocab_text,
            verbose=verbose,
        )
    raise ValueError(f"unknown tokenizer type: {tokenizer_type}")


def tokenizer_from_config(config: dict, verbose: bool = True):
    tokenizer_type = config["type"]
    if tokenizer_type == "char":
        return Tokenizer.from_config(config, verbose=verbose)
    if tokenizer_type == "bpe":
        return BPETokenizer.from_config(config, verbose=verbose)
    raise ValueError(f"unknown tokenizer type: {tokenizer_type}")

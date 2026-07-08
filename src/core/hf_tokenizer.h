#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

/// Reads tokenizer.json (HF tokenizers format) with byte-level BPE encoding.
/// Provides exact token matching with HuggingFace `PreTrainedTokenizerFast`.
class HFTokenizer {
public:
    HFTokenizer() = default;

    bool load(const std::string & path);
    std::vector<int32_t> encode(const std::string & text) const;
    int32_t token_to_id(const std::string & token) const;

private:
    std::vector<int32_t> bpe_encode_word(const std::string & word) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::vector<std::pair<std::string, int32_t>> special_tokens_;
};

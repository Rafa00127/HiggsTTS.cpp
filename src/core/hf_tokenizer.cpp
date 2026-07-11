#include "hf_tokenizer.h"
#include "json.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>

using json = nlohmann::json;

// ── byte-level BPE helpers ─────────────────────────────────────────────────

static const uint32_t * byte_to_cp_table() {
    static uint32_t table[256] = {};
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        for (int b = 0x21; b <= 0x7E; ++b) table[b] = (uint32_t)b;
        for (int b = 0xA1; b <= 0xAC; ++b) table[b] = (uint32_t)b;
        for (int b = 0xAE; b <= 0xFF; ++b) table[b] = (uint32_t)b;
        uint32_t extra = 0x0100;
        for (int b = 0x00; b <= 0x20; ++b) table[b] = extra++;
        for (int b = 0x7F; b <= 0xA0; ++b) table[b] = extra++;
        table[0xAD] = extra;
    }
    return table;
}

static std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80)        { s += (char)cp; }
    else if (cp < 0x800)  { s += (char)(0xC0 | (cp >> 6));  s += (char)(0x80 | (cp & 0x3F)); }
    else                  { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    return s;
}

static std::string to_byte_level(const std::string & raw) {
    const uint32_t * tbl = byte_to_cp_table();
    std::string result;
    for (unsigned char b : raw) result += cp_to_utf8(tbl[b]);
    return result;
}

static std::vector<std::string> utf8_chars(const std::string & text) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = 1;
        unsigned char c = (unsigned char)text[i];
        if      ((c & 0xF8) == 0xF0) len = 4;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xE0) == 0xC0) len = 2;
        if (i + len > text.size()) len = 1;
        chars.push_back(text.substr(i, len));
        i += len;
    }
    return chars;
}

// ── GPT-2 ByteLevel pre-tokenizer ────────────────────────────────────────────
// Matches the regex from tokenizer.json:
// (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// Implemented as a character-class scan to avoid std::regex dependency.

static bool pch_is_space(const std::string & ch) {
    if (ch.size() != 1) return false;
    char c = ch[0];
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Decode a single UTF-8 character to a Unicode code point.
static uint32_t decode_utf8_cp(const std::string & ch) {
    if (ch.empty()) return 0;
    unsigned char c = (unsigned char)ch[0];
    if (c < 0x80) return c;
    size_t len;
    uint32_t cp;
    if ((c & 0xE0) == 0xC0)      { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
    else return 0xFFFD;
    if (ch.size() < len) return 0xFFFD;
    for (size_t i = 1; i < len; i++)
        cp = (cp << 6) | ((unsigned char)ch[i] & 0x3F);
    return cp;
}

// Check if a code point is punctuation (not a letter).
// All ranges listed here are excluded from "letter" classification so they
// get split into their own pre-tokens in the GPT-2 ByteLevel pre-tokenizer.
static bool cp_is_punct(uint32_t cp) {
    return (cp >= 0x00A0 && cp <= 0x00BF)   // Latin-1 Supplement: NBSP, ¡, ¢…¿, · (middle dot)
        || (cp >= 0x2000 && cp <= 0x206F)   // General Punctuation: "" '' — – … ‹› †‡• etc.
        || (cp >= 0x3000 && cp <= 0x303F)   // CJK Symbols and Punctuation: 。、「」【】
        || (cp >= 0xFF01 && cp <= 0xFF0F)   // Fullwidth !../
        || (cp >= 0xFF1A && cp <= 0xFF20)   // Fullwidth :..@
        || (cp >= 0xFF3B && cp <= 0xFF40)   // Fullwidth [..`
        || (cp >= 0xFF5B && cp <= 0xFF5E)   // Fullwidth {..~
        || (cp >= 0xFF61 && cp <= 0xFF65);  // Halfwidth punctuation
}

static bool pch_is_letter(const std::string & ch) {
    if (ch.size() == 1) {
        unsigned char c = (unsigned char)ch[0];
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }
    // multi-byte UTF-8: exclude punctuation ranges, treat rest as letter
    return !cp_is_punct(decode_utf8_cp(ch));
}

static bool pch_is_digit(const std::string & ch) {
    if (ch.size() != 1) return false;
    return ch[0] >= '0' && ch[0] <= '9';
}

static int match_contraction_len(const std::vector<std::string> & chars, size_t i) {
    if (i >= chars.size() || chars[i] != "'") return 0;
    if (i + 1 >= chars.size()) return 0;
    std::string c1 = chars[i + 1];
    // ASCII lowercase
    if (c1.size() == 1 && c1[0] >= 'A' && c1[0] <= 'Z') c1[0] += 'a' - 'A';
    if (c1 == "s" || c1 == "t" || c1 == "m" || c1 == "d") return 2;
    if (i + 2 < chars.size()) {
        std::string c2 = chars[i + 2];
        if (c2.size() == 1 && c2[0] >= 'A' && c2[0] <= 'Z') c2[0] += 'a' - 'A';
        if (c1 == "r" && c2 == "e") return 3;
        if (c1 == "v" && c2 == "e") return 3;
        if (c1 == "l" && c2 == "l") return 3;
    }
    return 0;
}

static std::vector<std::string> pre_tokenize(const std::string & text) {
    auto chars = utf8_chars(text);
    std::vector<std::string> words;
    size_t i = 0;
    while (i < chars.size()) {
        // ── consume optional leading space (patterns 2/4's optional prefix) ──
        bool has_space = false;
        std::string space_str;
        if (pch_is_space(chars[i]) && !(chars[i] == "\r" || chars[i] == "\n")
            && i + 1 < chars.size() && !pch_is_space(chars[i + 1])) {
            has_space = true;
            space_str = chars[i];
            i++;
        }

        if (i >= chars.size()) {
            if (has_space) words.push_back(space_str);
            break;
        }

        // ── try contraction (alternation 1) ──
        int clen = match_contraction_len(chars, i);
        if (clen > 0) {
            std::string token = space_str;
            for (int k = 0; k < clen; k++) token += chars[i + (size_t)k];
            words.push_back(token);
            i += (size_t)clen;
            continue;
        }

        // ── classify current character ──
        if (pch_is_letter(chars[i])) {
            // pattern 2: [^\r\n\p{L}\p{N}]?\p{L}+
            std::string token = space_str;
            while (i < chars.size() && pch_is_letter(chars[i]))
                token += chars[i++];
            words.push_back(token);
        } else if (pch_is_digit(chars[i])) {
            // pattern 3: \p{N}  (single digit)
            words.push_back(space_str + chars[i]);
            i++;
        } else if (!pch_is_space(chars[i])) {
            // pattern 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
            std::string token = space_str;
            while (i < chars.size() && !pch_is_space(chars[i])
                   && !pch_is_letter(chars[i]) && !pch_is_digit(chars[i])) {
                token += chars[i++];
            }
            // absorb trailing \r\n (pattern 4 suffix & pattern 5)
            while (i < chars.size() && (chars[i] == "\r" || chars[i] == "\n"))
                token += chars[i++];
            words.push_back(token);
        } else {
            // patterns 6+7: \s+(?!\S) | \s+  (standalone / trailing whitespace)
            std::string token = space_str;
            while (i < chars.size() && pch_is_space(chars[i]))
                token += chars[i++];
            words.push_back(token);
        }
    }
    return words;
}

static std::vector<std::string> split_on_specials(
    const std::string & text,
    const std::vector<std::pair<std::string, int32_t>> & specials,
    std::vector<int32_t> & special_ids)
{
    struct Match { size_t pos; size_t len; int32_t id; };
    std::vector<Match> matches;
    for (const auto & sp : specials) {
        size_t pos = 0;
        while ((pos = text.find(sp.first, pos)) != std::string::npos) {
            matches.push_back({pos, sp.first.size(), sp.second});
            pos += sp.first.size();
        }
    }
    std::sort(matches.begin(), matches.end(),
        [](const Match & a, const Match & b) { return a.pos < b.pos || (a.pos == b.pos && a.len > b.len); });
    std::vector<Match> filtered;
    size_t last_end = 0;
    for (const auto & m : matches) {
        if (m.pos >= last_end) { filtered.push_back(m); last_end = m.pos + m.len; }
    }
    std::vector<std::string> segments;
    special_ids.clear();
    size_t pos = 0;
    for (const auto & m : filtered) {
        if (m.pos > pos) { segments.push_back(text.substr(pos, m.pos - pos)); special_ids.push_back(-1); }
        segments.push_back(text.substr(m.pos, m.len));
        special_ids.push_back(m.id);
        pos = m.pos + m.len;
    }
    if (pos < text.size()) { segments.push_back(text.substr(pos)); special_ids.push_back(-1); }
    return segments;
}

// ── BPE ─────────────────────────────────────────────────────────────────────

std::vector<int32_t> HFTokenizer::bpe_encode_word(const std::string & word) const {
    if (word.empty()) return {};
    const std::string bl = to_byte_level(word);

    auto it = vocab_.find(bl);
    if (it != vocab_.end()) return {it->second};

    std::vector<std::string> symbols = utf8_chars(bl);
    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_pos   = std::string::npos;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string pair = symbols[i] + symbols[i + 1];
            auto it2 = merge_rank_.find(pair);
            if (it2 != merge_rank_.end() && it2->second < best_rank) {
                best_rank = it2->second;
                best_pos  = i;
            }
        }
        if (best_pos == std::string::npos) break;
        symbols[best_pos] += symbols[best_pos + 1];
        symbols.erase(symbols.begin() + (long)best_pos + 1);
    }

    std::vector<int32_t> ids;
    for (const auto & sym : symbols) {
        auto it2 = vocab_.find(sym);
        if (it2 != vocab_.end()) ids.push_back(it2->second);
    }
    return ids;
}

// ── load ────────────────────────────────────────────────────────────────────

bool HFTokenizer::load(const std::string & path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[hf_tokenizer] failed to open: %s\n", path.c_str());
        return false;
    }
    json j;
    try { f >> j; }
    catch (const json::parse_error & e) {
        std::fprintf(stderr, "[hf_tokenizer] JSON parse error: %s\n", e.what());
        return false;
    }

    // added_tokens
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto & tok : j["added_tokens"]) {
            std::string content = tok.value("content", "");
            int32_t id = tok.value("id", -1);
            if (!content.empty() && id >= 0) {
                vocab_[content] = id;
                if (tok.value("special", false))
                    special_tokens_.push_back({content, id});
            }
        }
    }
    // model.vocab
    if (j.contains("model") && j["model"].contains("vocab")) {
        for (auto it2 = j["model"]["vocab"].begin(); it2 != j["model"]["vocab"].end(); ++it2) {
            int32_t id = it2.value().get<int32_t>();
            const std::string & key = it2.key();
            if (vocab_.find(key) == vocab_.end()) vocab_[key] = id;
        }
    }
    // model.merges
    if (j.contains("model") && j["model"].contains("merges")) {
        int32_t rank = 0;
        for (const auto & m : j["model"]["merges"]) {
            if (m.is_string()) {
                std::string s = m.get<std::string>();
                size_t sp = s.find(' ');
                if (sp != std::string::npos)
                    merge_rank_[s.substr(0, sp) + s.substr(sp + 1)] = rank++;
            } else if (m.is_array() && m.size() >= 2) {
                std::string a = m[0].get<std::string>();
                std::string b = m[1].get<std::string>();
                merge_rank_[a + b] = rank++;
            }
        }
    }

    std::sort(special_tokens_.begin(), special_tokens_.end(),
        [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    return true;
}

// ── encode ──────────────────────────────────────────────────────────────────

std::vector<int32_t> HFTokenizer::encode(const std::string & text) const {
    if (text.empty()) return {};

    std::vector<int32_t> special_ids;
    auto segments = split_on_specials(text, special_tokens_, special_ids);

    std::vector<int32_t> ids;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (special_ids[i] != -1) {
            ids.push_back(special_ids[i]);
        } else if (!segments[i].empty()) {
            auto words = pre_tokenize(segments[i]);
            for (const auto & w : words) {
                auto w_ids = bpe_encode_word(w);
                ids.insert(ids.end(), w_ids.begin(), w_ids.end());
            }
        }
    }
    return ids;
}

int32_t HFTokenizer::token_to_id(const std::string & token) const {
    auto it = vocab_.find(token);
    return (it != vocab_.end()) ? it->second : -1;
}

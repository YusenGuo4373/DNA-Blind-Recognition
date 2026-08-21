#include "include/ACEncode.h"
#include "include/FastaParser.h"
#include "single_include/nlohmann/json.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

std::mutex aeon_result_mutex;
std::mutex *res_lock = &aeon_result_mutex;

static uint64_t next_u64(uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

static string2stringVec read_motifs(const std::string &path) {
    std::ifstream input(path);
    nlohmann::json document = nlohmann::json::parse(input);
    string2stringVec motifs;
    for (const auto &entry : document["motif"].items()) {
        motifs[entry.key()] = entry.value().get<std::vector<std::string>>();
    }
    return motifs;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: dna_aeon_encode COUNT SEED CODEBOOK MOTIFS OUTPUT_FASTA\n";
        return 2;
    }
    const int count = std::stoi(argv[1]);
    uint64_t state = std::stoull(argv[2]);
    auto codewords = parseFasta(argv[3]);
    auto motifs = read_motifs(argv[4]);
    if (count <= 0 || codewords.empty()) return 3;
    const int codeword_length = static_cast<int>(codewords.begin()->size());
    ProbMap probability_model(codeword_length, false, codewords, motifs);
    auto frequency_dictionary = probability_model.freqDict();
    auto transition_dictionary = probability_model.createTransitionDict(frequency_dictionary);

    std::ofstream fasta(argv[5]);
    if (!fasta.good()) return 4;
    for (int item = 0; item < count; ++item) {
        std::string payload(40, '\0');
        for (char &value : payload) value = static_cast<char>(next_u64(state) & 0xffu);
        std::stringstream input(payload);
        BitInStream bits(input, 2); // official default: CRC marker every two bytes
        FreqTable frequencies(motifs, "", 0, false, codeword_length, transition_dictionary);
        frequencies.calcFreqs();
        std::stringstream encoded;
        inflating(frequencies, bits, encoded, 384);
        const std::string sequence = encoded.str();
        if (sequence.size() != 384) {
            std::cerr << "unexpected DNA-Aeon length " << sequence.size() << "\n";
            return 5;
        }
        for (char base : sequence) {
            if (base != 'A' && base != 'C' && base != 'G' && base != 'T') return 6;
        }
        fasta << ">dna_aeon_" << item << "\n" << sequence << "\n";
    }
    return 0;
}

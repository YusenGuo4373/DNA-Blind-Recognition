#include "nr3b.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

// DNAcode.h in upstream commit 86812c5 declares encode by value while the
// implementation takes const-reference. Declare the implemented ABI here.
void setcoderate(Int pattnumber, char *leftprimer, char *rightprimer);
void setdnaconstraints(Int window, Int maxgc, Int mingc, Int maxrun);
VecUchar encode(const VecUchar &message, Int len);
void releaseall();

static uint64_t next_u64(uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "usage: hedges_encode COUNT SEED OUTPUT_FASTA\n";
        return 2;
    }
    const int count = std::stoi(argv[1]);
    uint64_t state = std::stoull(argv[2]);
    std::ofstream output(argv[3]);
    if (!output.good() || count <= 0) return 3;

    // Pure HEDGES inner code: no fixed flanking primers, 384 encoded nucleotides.
    char leftprimer[] = "";
    char rightprimer[] = "";
    setcoderate(3, leftprimer, rightprimer); // rate 1/2
    setdnaconstraints(12, 8, 4, 4);

    static const char alphabet[] = {'A', 'C', 'G', 'T'};
    for (int item = 0; item < count; ++item) {
        VecUchar payload(40, Uchar(0));
        for (int i = 0; i < 40; ++i) payload[i] = Uchar(next_u64(state) & 0xffu);
        VecUchar encoded = encode(payload, 384);
        if (encoded.size() != 384) {
            std::cerr << "unexpected HEDGES length " << encoded.size() << "\n";
            return 4;
        }
        output << ">hedges_" << item << "\n";
        for (int i = 0; i < encoded.size(); ++i) {
            if (encoded[i] > 3) return 5;
            output << alphabet[encoded[i]];
        }
        output << "\n";
    }
    releaseall();
    return 0;
}

# Official inner-code adapters

These command-line wrappers call the upstream codec implementations without
changing the author's blind recognizer.

- `hedges_encode.cpp` calls HEDGES `encode()` at rate 1/2 with the biochemical
  constraints used by its official demo, requests 384-nt output, and passes
  empty left/right primers so the test contains only the HEDGES inner code.
- `dna_aeon_encode.cpp` calls DNA-Aeon's `Inflate`, `FreqTable`, `ProbMap`, and
  `BitInStream` with the official `cw_40_60_hp3` codebook and a CRC sync interval
  of two bytes. It deliberately bypasses NOREC4DNA, so the generated unknown is
  the DNA-Aeon inner code rather than its outer Raptor fountain code.

The DNA-Aeon GCC 8 build copies `ProbabilityEval.cpp` to the build directory and
replaces the single C++20 `std::string::ends_with` expression with its equivalent
`compare` expression. The upstream snapshot remains unmodified.

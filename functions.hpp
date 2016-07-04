#pragma once

template < int IMinusExtent, int IPlusExtent, int JMinusExtent, int JPlusExtent >
GT_FUNCTION bool is_in_domain(
    const int iblock_pos, const int jblock_pos, const unsigned int block_size_i, const unsigned int block_size_j) {
    return (iblock_pos >= IMinusExtent && iblock_pos < ((int)block_size_i + IPlusExtent) &&
            jblock_pos >= JMinusExtent && jblock_pos < ((int)block_size_j + JPlusExtent));
}


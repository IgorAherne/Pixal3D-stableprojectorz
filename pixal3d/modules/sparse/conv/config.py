SPCONV_ALGO = 'auto'                                # 'auto', 'implicit_gemm', 'native'
FLEX_GEMM_ALGO = 'masked_implicit_gemm_splitk'      # 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 2.0                       # Ratio of hashmap size to input size
LEAN_NEIGHBOR_CACHE = True                          # False: flex_gemm builds every neighbour cache itself.
# The lean cache (conv/lean_cache.py) keeps only what a forward pass reads, a third of the memory. It gives the
# same tables, so a wrong one would be a quietly different mesh rather than an error: this is the way back.

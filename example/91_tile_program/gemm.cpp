template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct GemmMultiD
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    __host__ __device__ void
    operator()(TileProgram& tp,
               const std::array<index_t, 2> a_m_k_lengths,
               const std::array<index_t, 2> a_m_k_strides,
               const std::array<index_t, 2> b_n_k_lengths,
               const std::array<index_t, 2> b_n_k_strides,
               const std::array<const std::array<index_t, 2>, NumDTensor> ds_m_n_lengths,
               const std::array<const std::array<index_t, 2>, NumDTensor> ds_m_n_strides,
               const std::array<index_t, 2> e_m_n_lengths,
               const std::array<index_t, 2> e_m_n_strides,
               //
               const T* p_a,
               const T* p_b,
               const std::array<const T*> p_ds,
               T* p_e)
    {
        using namespace ck;

        const auto b  = tp(make_naive_tensor(b_n_k_lengths, b_n_k_strides), p_b);
        const auto ds = tp(generate_tuple(
            [&](auto i) {
                return make_naive_tensor(ds_m_n_lengths[i], ds_m_n_strides[i], p_ds[i]),
            },
            Number<NumDTensor>{}));
        auto e        = tp(make_naive_tensor(e_m_n_lengths, e_m_n_strides), p_e);

        // divide problem
        const auto num_m = e_m_n_lengths[0];
        const auto num_n = e_m_n_lengths[1];

        const auto id_block = get_block_1d_id();

        const auto num_tile_m = num_gemmm / MPerTile;
        const auto num_tile_n = num_gemmn / NPerTile;

        const auto block2tile = tp(make_cluster_descriptor(make_tuple(num_tile_m, num_tile_n)));

        const auto id_tile = block2tile.CalculateBottonIndex(id_block);

        const auto id_tile_m = id_tile.At<0>();
        const auto id_tile_n = id_tile.At<1>();

        // A/B in DRAM
        // A/B DRAM layout is part of problem, not solution
#if 1
        // DO NOT let user know there is optimization on tensor transform on A/B DRAM tensor
        const auto a_dram_global = tp(make_naive_tensor(a_m_k_lengths, a_m_k_strides), p_a_dram);
        const auto b_dram_global = tp(make_naive_tensor(b_n_k_lengths, b_n_k_strides), p_b_dram);
#endif

        // A/B tile in LDS
        // A/B DRAM layout is part of solution
        ADataType* p_a_lds = shared_memmory.get_pointer(0);

        // [allow optimization] allow different LDS layouts
        constexpr auto a_lds_block =
            make_tensor(p_a_lds, {kMPerBlock, kKPerBlock}, a_lds_block_strategy);

        constexpr auto a_lds_byte = a_lds_block.get_num_of_byte();

        BDataType* p_b_lds = shared_memory.get_aligned_pointer(a_lds_byte);

        // [allow optimization] allow different LDS layouts
        constexpr auto b_lds_block =
            make_tensor({p_b_lds, kNPerBlock, kKPerBlock}, b_lds_block_strategy);

        // A/B copy
#if 0
        auto a_block_copy = make_copier(a_dram_global,
                                        a_lds_block,
                                        make_tuple(kMPerBlock, kKPerBlock),
                                        make_tuple(id_tile_m * kMPerBlock, 0),
                                        a_block_copy_strategy);

        auto b_block_copy = make_copier(b_dram_global,
                                        b_lds_block,
                                        make_tuple(kNPerBlock, kKPerBlock),
                                        make_tuple(id_tile_n * kNPerBlock, 0),
                                        b_block_copy_strategy);
#else
        auto window_a_dram = make_window(a_dram_global,
                                         {MPerTile, KPerTile},
                                         {id_tile_m * MPerTile, id_tile_k * KPerTile},
                                         a_dram_window_map_strategy);

        auto window_a_block =
            make_window(a_lds_block, {NPerTile, KPerTile}, {0, 0}, a_lds_window_map_strategy);

#endif

#if 1
        // block GEMM
        // operation-based syntax: per-operation solution strategy
        auto block_gemm = make_block_gemm(a_lds_block, b_lds_block, block_gemm_strategy);
#endif

        // Distributed C in VGPR
#if 1
        // C layout is decided alone
        // C should be distributed,
        auto c_vgpr_block =
            make_distributed_tensor({kMPerBlock, kNPerBlock}, c_vgpr_block_strategy);
#elif 0
        // C layout is decided by block GEMM
        auto c_vgpr_block = block_gemm.get_c_vgpr_block();
#endif

        for(index_t k = 0; k < K; k += kKPerBlock)
        {
            auto a_vgpr_block_tmp = load(window_a_dram, a_dram_load_strategy);
            auto b_vgpr_block_tmp = load(window_b_dram, b_dram_load_strategy);

            auto a_vpgr_block = elementwise_op(a_vgpr_block_tmp, a_element_op);
            auto b_vpgr_block = elementwise_op(b_vgpr_block_tmp, b_element_op);

            copy(a_vgpr_block, a_lds_block, a_lds_store_strategy);
            copy(b_vgpr_block, b_lds_block, b_lds_store_strategy);

            block_sync_lds();

            dot_product_accumulate(c_vgpr_block, a_lds_block, b_lds_block);

            block_sync_lds();

            window_a_dram += {0, kKPerBlock};
            window_b_dram += {0, kKPerBlock};
        }

        auto p_c_lds = xxx;

        auto c_lds = make_tensor(p_c_lds, xxxxxx);

        auto window_c_vgpr =
            make_window(c_vgpr, {kMPerShuffle, kNPerShuffle}, {0, 0}, c_vgpr_window_strategy);

        auto window_c_lds =
            make_window(c_lds, {kMPerShuffle, kNPerShuffle}, {0, 0}, c_lds_window_strategy);

        auto window_d_dram = make_window(d_dram_global,
                                         {kMPerShuffle, kNPerShuffle},
                                         {id_tile_m * kMPerTile, id_tile_n * kNPerTile},
                                         d_dram_window_strategy);

        auto window_e_dram = make_window(e_dram_global,
                                         {kMPerShuffle, kNPerShuffle},
                                         {id_tile_m * kMPerTile, id_tile_n * kNPerTile},
                                         e_dram_window_strategy);

        for(m = 0; m < kMPerBlock; m += kMPerShuffle)
        {
            for(n = 0; n < kNPerBlock; n += kNPerShuffle)
            {
                // write C into LDS for shuffle
                copy(window_c_vgpr, window_c_lds, c_lds_store_strategy);

                // load C from LDS to complete shuffle
                auto c_vgpr_slice_shuffled = load(window_c_lds, c_lds_load_strategy);

                // load D from dram
                auto d_vgpr_block_slice = load(window_d_dram, d_dram_load_strategy);

                // element wise op
                // [Question] need to gurantee it always function
                //   1. C/D should have same layout, how to gurantee?
                //   2. if C/D have different layout, then need to do shuffle
                //   3. if C/D have different layout, what should E layout be?
                auto e_vgpr_block_slice =
                    elementwise_op(c_vgpr_block_slice, d_vgpr_block_slice, cd_elementwise_op);

                // write E into dram
                copy(e_vgpr_block_slice, window_e_dram, e_dram_store_strategy);
            }
        }
    }
};

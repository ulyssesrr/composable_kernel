#include "tile_program.hpp"

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

#include "ck/library/utility/device_memory.hpp"

// program
template <ck::index_t NDimSpatial, typename ALayout>
struct Im2Col
{
    __host__ __device__ void
    operator()(TileProgram& tp,
               const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
               const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
               const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
               const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
               const std::array<index_t, NDimSpatial>& conv_filter_strides,
               const std::array<index_t, NDimSpatial>& conv_filter_dilations,
               const std::array<index_t, NDimSpatial>& input_left_pads,
               const std::array<index_t, NDimSpatial>& input_right_pads,
               //
               const std::array<index_t, 2> a_gemmg_gemmm_gemmk_lengths,
               const std::array<index_t, 2> a_gemmg_gemmm_gemmk_strides,
               //
               const T* p_a_img,
               T* p_a_mtx)
    {
        using namespace ck;

        const auto a_src_desc =
            tp(TransformConvFwdToGemm<NDimSpatial, ConvolutionForwardSpecialization::Default>::
                   template MakeADescriptor_M_K<ALayout>(a_g_n_c_wis_lengths,
                                                         a_g_n_c_wis_strides,
                                                         b_g_k_c_xs_lengths,
                                                         b_g_k_c_xs_strides,
                                                         c_g_n_k_wos_lengths,
                                                         c_g_n_k_wos_strides,
                                                         conv_filter_strides,
                                                         conv_filter_dilations,
                                                         input_left_pads,
                                                         input_right_pads));

        const auto a_dst_desc =
            tp(make_naive_tensor_descriptor(a_gemmm_gemmk_lengths, a_gemmm_gemmk_strides));

        const auto a_src = make_tensor(a_src_desc, p_a_img);

        const auto a_dst = make_tensor(a_dst_desc, p_a_mtx);

        const auto num_gemmg = a_gemmg_gemmm_gemmk_c_wis_lengths[0];
        const auto num_gemmm = a_gemmg_gemmm_gemmk_c_wis_lengths[1];
        const auto num_gemmk = a_gemmg_gemmm_gemmk_c_wis_lengths[2];

        const auto id_block = get_block_1d_id();

        const auto num_tile_m = num_gemmm / MPerTile;
        const auto num_tile_k = num_gemmk / KPerTile;

        const auto block2tile = tp(make_cluster_descriptor(make_tuple(num_tile_m, num_tile_k)));

        const auto id_tile = block2tile.CalculateBottonIndex(id_block);

        const auto id_tile_m = id_tile.At<0>();
        const auto id_tile_k = id_tile.At<1>();

#if 0
        // operation-based syntax: per-oeration solution strategy
        // operation here is data movement
        auto copier = make_copier(a_src,
                                  a_dst,
                                  make_tuple(1, MPerTile, KPerTile),
                                  make_tuple(0, id_tile_m * MPerTile, id_tile_k * KPerTile),
                                  copy_strategy);

        for(ck::index_t id_gemmg = 0; id_gemmg < num_gemmg; id_gemmg++)
        {
            copier();

            copier.move_src_window(make_tuple(1, 0, 0));
            copier.move_dst_window(make_tuple(1, 0, 0));
        }
#else
        // data-based syntax: per-data solution strategy
        auto window_a_src = make_window(a_src,
                                        make_tuple(1, MPerTile, KPerTile),
                                        make_tuple(0, id_tile_m * MPerTile, id_tile_k * KPerTile),
                                        a_src_window_map_strategy);

        auto window_a_dst = make_window(a_dst,
                                        make_tuple(1, MPerTile, KPerTile),
                                        make_tuple(0, id_tile_m * MPerTile, id_tile_k * KPerTile),
                                        a_dst_window_map_strategy);

        for(ck::index_t id_gemmg = 0; id_gemmg < num_gemmg; id_gemmg++)
        {
            copy(window_a_src, window_a_dst, a_copy_strategy);

            window_a_src += make_tuple(1, 0, 0);
            window_a_dst += make_tuple(1, 0, 0);
        }
#endif
    }
};

int main()
{
    ck::index_t NumDimSpatial = 2;
    ck::index_t G             = 32;
    ck::index_t N             = 256;
    ck::index_t K             = 192;
    ck::index_t C             = 192;
    ck::index_t Y             = 3;
    ck::index_t X             = 3;
    ck::index_t Hi            = 28;
    ck::index_t Wi            = 28;
    ck::index_t Ho            = 28;
    ck::index_t Wo            = 28;

    std::array<ck::index_t, NumDimSpatial + 3> in_lengths{G, N, Hi, Wi, C};
    std::array<ck::index_t, NumDimSpatial + 3> in_strides{0, 0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 3> wei_lengths{G, K, Y, X, C};
    std::array<ck::index_t, NumDimSpatial + 3> wei_strides{0, 0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 3> out_lengths{G, N, Ho, Wo, K};
    std::array<ck::index_t, NumDimSpatial + 3> out_strides{0, 0, 0, 0, 1};

    std::partial_sum(rbegin(in_lengths),
                     std::prev(rend(in_lengths)),
                     std::next(rbegin(in_strides)),
                     std::multiplies<>{});
    std::partial_sum(rbegin(wei_lengths),
                     std::prev(rend(wei_lengths)),
                     std::next(rbegin(wei_strides)),
                     std::multiplies<>{});
    std::partial_sum(rbegin(out_lengths),
                     std::prev(rend(out_lengths)),
                     std::next(rbegin(out_strides)),
                     std::multiplies<>{});

    // transpose GNHWC/GKYXC/GNHWK to GNCHW/GKCYX/GNCHW
    std::rotate(
        rbegin(in_lengths), std::next(rbegin(in_lengths)), std::next(rbegin(in_lengths), 3));
    std::rotate(
        rbegin(in_strides), std::next(rbegin(in_strides)), std::next(rbegin(in_strides), 3));
    std::rotate(
        rbegin(wei_lengths), std::next(rbegin(wei_lengths)), std::next(rbegin(wei_lengths), 3));
    std::rotate(
        rbegin(wei_strides), std::next(rbegin(wei_strides)), std::next(rbegin(wei_strides), 3));
    std::rotate(
        rbegin(out_lengths), std::next(rbegin(out_lengths)), std::next(rbegin(out_lengths), 3));
    std::rotate(
        rbegin(out_strides), std::next(rbegin(out_strides)), std::next(rbegin(out_strides), 3));

    std::array<ck::index_t, NumDimSpatial> filter_strides{1, 1};
    std::array<ck::index_t, NumDimSpatial> filter_dilations{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1};

    // matrix
    std::array<ck::index_t, NumDimSpatial + 3> in_mtx_lengths{G, G * Ho * Wo, C * Y * X};
    std::array<ck::index_t, NumDimSpatial + 3> in_mtx_strides{0, 0, 1};

    std::partial_sum(rbegin(in_mtx_lengths),
                     std::prev(rend(in_mtx_lengths)),
                     std::next(rbegin(in_mtx_strides)),
                     std::multiplies<>{});

    DeviceMem in(sizeof(InDataType) * G * N * Hi * Wi * C);
    DeviceMem in_mtx(sizeof(InDataType) * G * N * Ho * Wo * C * Y * X);

    launch(HelloWorld{},
           1,
           1,
           in_lengths,
           in_strides,
           wei_lengths,
           wei_strides,
           out_lengths,
           out_strides,
           filter_strides,
           filter_dilations,
           input_left_pads,
           input_right_pads,
           //
           in_mtx_lengths,
           in_mtx_strides,
           //
           in.GetDeviceBuffer(),
           in_mtx.GetDeviceBuffer());

    return 0;
}

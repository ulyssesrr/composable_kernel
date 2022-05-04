	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.protected	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_ ; -- Begin function _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
	.globl	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
	.p2align	8
	.type	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_,@function
_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_: ; @_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
; %bb.0:
	s_mov_b64 s[66:67], s[2:3]
	s_mov_b64 s[64:65], s[0:1]
	s_add_u32 s64, s64, s7
	s_load_dwordx2 s[12:13], s[4:5], 0x0
	s_load_dwordx2 s[16:17], s[4:5], 0x8
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	s_load_dwordx2 s[2:3], s[4:5], 0x24
	s_load_dword s33, s[4:5], 0x48
	s_load_dword s10, s[4:5], 0x50
	s_load_dword s11, s[4:5], 0x58
	s_load_dwordx2 s[42:43], s[4:5], 0x6c
	s_load_dword s56, s[4:5], 0x84
	s_load_dwordx4 s[20:23], s[4:5], 0x98
	s_load_dwordx4 s[24:27], s[4:5], 0xac
	s_load_dwordx2 s[28:29], s[4:5], 0xbc
	s_load_dwordx2 s[30:31], s[4:5], 0xd4
	s_load_dwordx2 s[34:35], s[4:5], 0xe4
	s_load_dwordx2 s[36:37], s[4:5], 0x114
	s_load_dwordx2 s[38:39], s[4:5], 0x120
	s_load_dwordx2 s[40:41], s[4:5], 0x12c
	s_load_dwordx2 s[0:1], s[4:5], 0x13c
	s_load_dwordx2 s[14:15], s[4:5], 0x148
	s_load_dwordx2 s[18:19], s[4:5], 0x154
	s_load_dword s57, s[4:5], 0x16c
	s_load_dword s58, s[4:5], 0x180
	s_load_dword s7, s[4:5], 0x18c
	s_waitcnt lgkmcnt(0)
	s_load_dword s23, s[4:5], 0x1b0
	s_load_dword s59, s[4:5], 0x1c4
	s_load_dword s60, s[4:5], 0x1d4
	s_load_dwordx4 s[44:47], s[4:5], 0x1e0
	s_load_dwordx4 s[48:51], s[4:5], 0x1f4
	s_load_dwordx4 s[52:55], s[4:5], 0x208
	s_addc_u32 s65, s65, 0
	v_lshrrev_b32_e32 v1, 5, v0
	v_lshrrev_b32_e32 v33, 7, v0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s4, s51, s6
	s_add_i32 s4, s6, s4
	s_lshr_b32 s4, s4, s55
	s_mul_i32 s5, s4, s47
	s_sub_i32 s5, s6, s5
	s_mul_hi_u32 s6, s4, s50
	s_add_i32 s6, s4, s6
	s_lshr_b32 s6, s6, s54
	s_mul_i32 s46, s6, s46
	s_sub_i32 s4, s4, s46
	s_mul_hi_u32 s46, s6, s49
	s_add_i32 s46, s6, s46
	s_lshr_b32 s46, s46, s53
	s_mul_i32 s45, s46, s45
	s_sub_i32 s6, s6, s45
	s_mul_hi_u32 s45, s46, s48
	s_add_i32 s45, s46, s45
	s_lshr_b32 s45, s45, s52
	v_mad_i32_i24 v17, v33, -4, v1
	s_mul_i32 s43, s45, s43
	v_add_u32_e32 v50, s43, v17
	s_mul_i32 s44, s45, s44
	v_mul_hi_u32 v2, v50, s10
	s_sub_i32 s44, s46, s44
	s_mul_i32 s6, s6, s60
	s_add_i32 s5, s5, s6
	s_mul_i32 s6, s44, s59
	s_movk_i32 s44, 0xffe0
	s_add_i32 s6, s6, s4
	v_mad_i32_i24 v24, v1, s44, v0
	s_lshl_b32 s4, s6, 8
	s_lshl_b32 s5, s5, 8
	v_lshlrev_b32_e32 v1, 3, v24
	v_add_u32_e32 v2, v50, v2
	v_lshrrev_b32_e32 v18, s11, v2
	v_add_u32_e32 v3, s4, v1
	v_add_u32_e32 v1, s5, v1
	s_mul_i32 s45, s45, s57
	v_mul_lo_u32 v2, v18, s33
	v_mul_hi_u32 v4, v1, s15
	v_add_u32_e32 v51, s45, v17
	v_mul_hi_u32 v5, v51, s39
	v_sub_u32_e32 v20, v50, v2
	v_add_u32_e32 v2, v1, v4
	v_lshrrev_b32_e32 v2, s19, v2
	v_add_u32_e32 v5, v51, v5
	v_mul_hi_u32 v4, v2, s14
	v_lshrrev_b32_e32 v5, s41, v5
	v_mul_hi_u32 v7, v5, s38
	v_mul_lo_u32 v9, v5, s37
	v_add_u32_e32 v4, v2, v4
	v_lshrrev_b32_e32 v4, s18, v4
	v_add_u32_e32 v7, v5, v7
	v_mul_lo_u32 v8, v4, s0
	v_lshrrev_b32_e32 v52, s40, v7
	v_mul_lo_u32 v7, v52, s36
	v_sub_u32_e32 v53, v51, v9
	v_sub_u32_e32 v8, v2, v8
	v_mul_lo_u32 v4, v4, s30
	v_sub_u32_e32 v54, v5, v7
	v_mul_lo_u32 v5, v8, s34
	v_mul_lo_u32 v7, v53, s35
	v_mul_lo_u32 v8, v54, s31
	v_lshlrev_b32_e32 v19, 2, v33
	v_lshl_or_b32 v6, v18, 3, v19
	v_add_u32_e32 v55, v7, v5
	v_mul_lo_u32 v6, v6, s2
	v_mul_lo_u32 v10, v20, s3
	v_mul_lo_u32 v2, v2, s1
	v_add_u32_e32 v56, v8, v4
	v_subrev_u32_e32 v4, s28, v55
	v_lshl_or_b32 v9, v52, 3, v19
	v_subrev_u32_e32 v5, s25, v56
	v_mul_lo_u32 v4, v4, s22
	v_mul_lo_u32 v7, v9, s20
	v_mul_lo_u32 v5, v5, s21
	v_add3_u32 v3, v3, v6, v10
	v_sub_u32_e32 v1, v1, v2
	v_add_u32_e32 v1, v1, v4
	v_add_u32_e32 v9, s2, v3
	v_add3_u32 v21, v1, v7, v5
	v_lshlrev_b32_e32 v5, 1, v9
	v_add_u32_e32 v9, s2, v9
	s_lshl_b32 s14, s56, 1
	s_mov_b32 s15, 0x20000
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v22, 1, v9
	v_add_u32_e32 v23, s2, v9
	buffer_load_dwordx4 v[1:4], v1, s[12:15], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[5:8], v5, s[12:15], 0 offen
	v_lshlrev_b32_e32 v25, 1, v23
	buffer_load_dwordx4 v[9:12], v22, s[12:15], 0 offen
	buffer_load_dwordx4 v[13:16], v25, s[12:15], 0 offen
	s_sub_i32 s27, s27, s29
	v_cmp_le_i32_e32 vcc, s28, v55
	v_cmp_gt_i32_e64 s[0:1], s27, v55
	s_sub_i32 s24, s24, s26
	s_and_b64 s[44:45], vcc, s[0:1]
	v_cmp_le_i32_e32 vcc, s25, v56
	v_cmp_gt_i32_e64 s[0:1], s24, v56
	s_and_b64 s[0:1], vcc, s[0:1]
	s_brev_b32 s26, -2
	v_mov_b32_e32 v22, s26
	s_and_b64 s[0:1], s[44:45], s[0:1]
	v_cndmask_b32_e64 v22, v22, 0, s[0:1]
	v_lshl_add_u32 v34, v21, 1, v22
	v_add_u32_e32 v21, s20, v21
	s_lshl_b32 s18, s58, 1
	s_mov_b32 s19, s15
	v_lshl_add_u32 v35, v21, 1, v22
	v_add_u32_e32 v21, s20, v21
	buffer_load_dwordx4 v[25:28], v34, s[16:19], 0 offen
	buffer_load_dwordx4 v[29:32], v35, s[16:19], 0 offen
	v_lshl_add_u32 v34, v21, 1, v22
	v_add_u32_e32 v57, s20, v21
	v_lshl_add_u32 v21, v57, 1, v22
	buffer_load_dwordx4 v[36:39], v34, s[16:19], 0 offen
	buffer_load_dwordx4 v[40:43], v21, s[16:19], 0 offen
	v_lshlrev_b32_e32 v34, 5, v33
	s_movk_i32 s0, 0x880
	s_movk_i32 s1, 0x44
	s_movk_i32 s29, 0x80
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a204, 0
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	
             v_pack_b32_f16 v44, v1, v5 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v46, v1, v5, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v45, v9, v13 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v47, v9, v13, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v2, v6 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v2, v6, op_sel:[1, 1] 
             
	;;#ASMEND
	v_and_b32_e32 v1, 63, v0
	v_and_b32_e32 v2, 32, v0
	v_sub_u32_e32 v1, v1, v2
	v_add_u32_e32 v58, v1, v34
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v10, v14 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v10, v14, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v3, v7 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v48, v3, v7, op_sel:[1, 1] 
             
	;;#ASMEND
	v_ashrrev_i16_e32 v3, 15, v58
	v_lshrrev_b16_e32 v3, 13, v3
	v_add_u16_e32 v3, v58, v3
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v11, v15 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v49, v11, v15, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v4, v8 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v4, v8, op_sel:[1, 1] 
             
	;;#ASMEND
	v_ashrrev_i16_e32 v4, 3, v3
	v_and_b32_e32 v3, -8, v3
	v_lshrrev_b32_e32 v2, 4, v0
	v_sub_u16_e32 v3, v58, v3
	v_and_b32_e32 v2, 2, v2
	v_bfe_i32 v59, v4, 0, 16
	v_bfe_i32 v60, v3, 0, 16
	v_mul_u32_u24_e32 v2, s0, v2
	v_mul_i32_i24_e32 v3, s1, v59
	v_lshlrev_b32_e32 v4, 3, v60
	v_add3_u32 v61, v3, v2, v4
	v_lshrrev_b32_e32 v3, 6, v0
	v_mad_i32_i24 v3, v33, -2, v3
	v_lshl_add_u32 v35, v3, 5, v1
	v_ashrrev_i32_e32 v1, 31, v35
	v_lshrrev_b32_e32 v1, 29, v1
	v_add_u32_e32 v1, v35, v1
	v_add_u32_e32 v3, 4, v50
	v_ashrrev_i32_e32 v62, 3, v1
	v_mul_hi_u32 v4, v3, s10
	v_mul_lo_u32 v15, v62, s1
	v_and_b32_e32 v1, -8, v1
	v_sub_u32_e32 v63, v35, v1
	v_lshlrev_b32_e32 v1, 3, v63
	v_add_u32_e32 v4, v3, v4
	v_add3_u32 v64, v15, v2, v1
	v_add_u32_e32 v1, 4, v51
	v_lshrrev_b32_e32 v22, s11, v4
	v_mul_hi_u32 v2, v1, s39
	v_mul_lo_u32 v4, v22, s33
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v12, v16 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v12, v16, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v2, v1, v2
	v_sub_u32_e32 v21, v3, v4
	v_sub_u32_e32 v4, v22, v18
	v_lshrrev_b32_e32 v2, s41, v2
	v_sub_u32_e32 v3, v21, v20
	v_lshl_add_u32 v4, v4, 3, -3
	v_mul_hi_u32 v15, v2, s38
	v_mul_lo_u32 v4, v4, s2
	v_mul_lo_u32 v3, v3, s3
	v_mul_lo_u32 v16, v17, s0
	v_add_u32_e32 v15, v2, v15
	v_lshrrev_b32_e32 v18, s40, v15
	v_add3_u32 v23, v3, v4, v23
	v_mul_lo_u32 v3, v2, s37
	v_mul_lo_u32 v15, v18, s36
	v_or_b32_e32 v4, v16, v19
	s_movk_i32 s0, 0x4400
	v_sub_u32_e32 v20, v1, v3
	v_sub_u32_e32 v19, v2, v15
	v_sub_u32_e32 v2, v20, v53
	v_sub_u32_e32 v3, v19, v54
	v_mul_lo_u32 v15, v2, s35
	v_mul_lo_u32 v16, v3, s31
	v_sub_u32_e32 v2, v18, v52
	v_lshl_add_u32 v17, v2, 3, -3
	v_mul_lo_u32 v52, v24, s1
	v_add_u32_e32 v2, v15, v55
	v_mul_lo_u32 v17, v17, s20
	v_mul_lo_u32 v15, v15, s22
	v_add_u32_e32 v3, v16, v56
	v_mul_lo_u32 v16, v16, s21
	v_add_lshl_u32 v4, v4, v52, 1
	v_add_u32_e32 v15, v17, v15
	ds_write2_b64 v4, v[44:45], v[46:47] offset1:2
	ds_write2_b64 v4, v[5:6], v[9:10] offset0:4 offset1:6
	ds_write2_b64 v4, v[13:14], v[48:49] offset0:8 offset1:10
	ds_write2_b64 v4, v[7:8], v[11:12] offset0:12 offset1:14
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v25, v29 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v25, v29, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v36, v40 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v36, v40, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v5, 0x4000, v4
	v_add3_u32 v24, v15, v16, v57
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v26, v30 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v26, v30, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v37, v41 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v37, v41, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v15, v27, v31 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v25, v27, v31, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v16, v38, v42 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v26, v38, v42, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v27, v28, v32 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v29, v28, v32, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v28, v39, v43 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v30, v39, v43, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v5, v[7:8], v[9:10] offset0:128 offset1:130
	ds_write2_b64 v5, v[11:12], v[13:14] offset0:132 offset1:134
	ds_write2_b64 v5, v[15:16], v[25:26] offset0:136 offset1:138
	ds_write2_b64 v5, v[27:28], v[29:30] offset0:140 offset1:142
	v_add_u32_e32 v8, 64, v35
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshrrev_b32_e32 v9, 29, v9
	v_add_u32_e32 v9, v8, v9
	v_ashrrev_i32_e32 v10, 3, v9
	v_sub_u32_e32 v10, v10, v62
	v_add_u32_e32 v6, s0, v4
	v_lshl_add_u32 v5, v64, 1, s0
	s_mov_b32 s0, 0xffffff8
	v_mul_lo_u32 v10, v10, s1
	v_and_b32_e32 v9, s0, v9
	v_sub_u32_e32 v8, v8, v9
	v_sub_u32_e32 v8, v8, v63
	v_add_u32_e32 v9, s29, v35
	v_lshl_add_u32 v8, v8, 3, v10
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshrrev_b32_e32 v10, 29, v10
	v_add_u32_e32 v10, v9, v10
	v_ashrrev_i32_e32 v11, 3, v10
	v_sub_u32_e32 v11, v11, v62
	v_mul_lo_u32 v11, v11, s1
	v_and_b32_e32 v10, s0, v10
	v_sub_u32_e32 v9, v9, v10
	v_sub_u32_e32 v9, v9, v63
	v_lshl_add_u32 v9, v9, 3, v11
	s_movk_i32 s0, 0xc0
	v_lshl_add_u32 v10, v9, 1, v5
	v_add_u32_e32 v9, s0, v35
	v_ashrrev_i32_e32 v11, 31, v9
	v_lshrrev_b32_e32 v11, 29, v11
	v_add_u32_e32 v11, v9, v11
	v_ashrrev_i32_e32 v12, 3, v11
	v_sub_u32_e32 v12, v12, v62
	v_mul_lo_u32 v12, v12, s1
	v_and_b32_e32 v11, 0xffffff8, v11
	v_sub_u32_e32 v9, v9, v11
	v_add_u32_e32 v13, s29, v58
	v_sub_u32_e32 v9, v9, v63
	v_lshrrev_b32_e32 v13, 3, v13
	v_lshl_add_u32 v9, v9, 3, v12
	v_sub_u32_e32 v13, v13, v59
	v_lshl_add_u32 v11, v9, 1, v5
	v_add_u32_e32 v9, 64, v58
	v_mul_lo_u32 v14, v13, s1
	v_add_u32_e32 v13, s0, v58
	v_lshrrev_b32_e32 v9, 3, v9
	v_lshrrev_b32_e32 v13, 3, v13
	v_sub_u32_e32 v9, v9, v59
	v_sub_u32_e32 v13, v13, v59
	v_mul_lo_u32 v9, v9, s1
	v_mul_lo_u32 v15, v13, s1
	v_and_b32_e32 v12, 7, v58
	v_sub_u32_e32 v12, v12, v60
	v_lshl_add_u32 v16, v12, 3, v61
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a255, 0
	v_add_lshl_u32 v13, v16, v9, 1
	v_add_lshl_u32 v12, v16, v14, 1
	v_add_lshl_u32 v9, v16, v15, 1
	v_add_u32_e32 v14, 8, v51
	v_add_u32_e32 v15, 8, v50
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a240, 0
	s_mov_b32 s43, 0
	s_mov_b32 s4, s39
	v_lshlrev_b32_e32 v7, 1, v61
	v_lshl_add_u32 v8, v8, 1, v5
	s_add_i32 s29, s42, -4
	s_sub_i32 s30, 0, s37
	s_sub_i32 s33, 0, s33
	s_movk_i32 s34, 0x1000
	v_mov_b32_e32 v16, v15
	v_mov_b32_e32 v17, v14
BB0_1:                                  ; %_ZZN2ck22move_tensor_coordinateINS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS2_IJiiiEEELb0EEENS3_INS2_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESB_NS_23Merge_v2_magic_divisionINS2_IJiiEEEEESB_NSA_IS7_EENS3_ISD_Lb0EEESB_SF_EEENS2_IJNS_8SequenceIJLi0EEEENSI_IJLi1EEEENSI_IJLi2EEEENSI_IJLi3EEEENSI_IJLi4ELi6EEEENSI_IJLi7EEEENSI_IJLi5EEEENSI_IJLi8EEEENSI_IJLi9EEEENSI_IJLi10EEEEEEENS2_IJNSI_IJLi1ELi2ELi3EEEENSI_IJLi4ELi5EEEENSI_IJLi6EEEESO_SQ_SR_SS_NSI_IJLi11ELi12EEEENSI_IJLi13EEEENSI_IJLi14EEEEEEENSI_IJLi11ELi12ELi13ELi14EEEEiEENS_16TensorCoordinateILi15EKS11_EENS_20TensorCoordinateStepILi10ELi4ENSI_IJLi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0EEEEEEEEvRKT_RT0_RKT1_ENKUlS19_E_clINS6_IiLi9EEEEEDaS19_.exit.i.i.i.i.i333.i
                                        ; =>This Inner Loop Header: Depth=1
	v_cmp_le_i32_e32 vcc, s28, v2
	v_cmp_gt_i32_e64 s[0:1], s27, v2
	v_lshlrev_b32_e32 v25, 1, v23
	v_add_u32_e32 v23, s2, v23
	s_and_b64 s[44:45], vcc, s[0:1]
	v_cmp_le_i32_e32 vcc, s25, v3
	v_cmp_gt_i32_e64 s[0:1], s24, v3
	v_lshlrev_b32_e32 v29, 1, v23
	v_add_u32_e32 v23, s2, v23
	s_and_b64 s[0:1], vcc, s[0:1]
	v_lshlrev_b32_e32 v36, 1, v23
	v_add_u32_e32 v23, s2, v23
	s_and_b64 s[0:1], s[0:1], s[44:45]
	v_mov_b32_e32 v44, s26
	v_lshlrev_b32_e32 v40, 1, v23
	v_cndmask_b32_e64 v56, v44, 0, s[0:1]
	buffer_load_dwordx4 v[25:28], v25, s[12:15], 0 offen
	v_lshl_add_u32 v44, v24, 1, v56
	buffer_load_dwordx4 v[29:32], v29, s[12:15], 0 offen
	v_add_u32_e32 v24, s20, v24
	buffer_load_dwordx4 v[36:39], v36, s[12:15], 0 offen
	v_lshl_add_u32 v48, v24, 1, v56
	buffer_load_dwordx4 v[40:43], v40, s[12:15], 0 offen
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[60:63], v7 offset1:1
	v_add_u32_e32 v24, s20, v24
	v_lshl_add_u32 v52, v24, 1, v56
	v_add_u32_e32 v24, s20, v24
	v_lshl_add_u32 v56, v24, 1, v56
	v_add_u32_e32 v64, s34, v7
	buffer_load_dwordx4 v[44:47], v44, s[16:19], 0 offen
	v_add_u32_e32 v72, s34, v5
	buffer_load_dwordx4 v[48:51], v48, s[16:19], 0 offen
	v_add_u32_e32 v80, s34, v8
	buffer_load_dwordx4 v[52:55], v52, s[16:19], 0 offen
	v_add_u32_e32 v88, s34, v10
	buffer_load_dwordx4 v[56:59], v56, s[16:19], 0 offen
	ds_read2_b64 v[64:67], v64 offset0:32 offset1:33
	ds_read2_b64 v[68:71], v5 offset1:1
	ds_read2_b64 v[76:79], v8 offset1:1
	ds_read2_b64 v[84:87], v10 offset1:1
	ds_read2_b64 v[92:95], v11 offset1:1
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8f16 a[192:207], v[60:61], v[68:69], a[192:207]
	v_add_u32_e32 v96, s34, v11
	ds_read2_b64 v[72:75], v72 offset0:32 offset1:33
	ds_read2_b64 v[80:83], v80 offset0:32 offset1:33
	ds_read2_b64 v[88:91], v88 offset0:32 offset1:33
	ds_read2_b64 v[96:99], v96 offset0:32 offset1:33
	v_mul_hi_u32 v101, s10, v16
	v_mul_hi_u32 v100, s4, v17
	v_add_u32_e32 v1, 4, v1
	v_add_u32_e32 v17, 4, v17
	v_add_u32_e32 v16, 4, v16
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8f16 a[224:239], v[60:61], v[76:77], a[224:239]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8f16 a[208:223], v[60:61], v[84:85], a[208:223]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8f16 a[176:191], v[60:61], v[92:93], a[176:191]
	v_mfma_f32_32x32x8f16 a[192:207], v[62:63], v[70:71], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[62:63], v[78:79], a[224:239]
	v_mfma_f32_32x32x8f16 a[208:223], v[62:63], v[86:87], a[208:223]
	v_mfma_f32_32x32x8f16 a[176:191], v[62:63], v[94:95], a[176:191]
	ds_read2_b64 v[60:63], v13 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[160:175], v[60:61], v[68:69], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[60:61], v[76:77], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[60:61], v[84:85], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[60:61], v[92:93], a[112:127]
	v_mfma_f32_32x32x8f16 a[192:207], v[64:65], v[72:73], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[64:65], v[80:81], a[224:239]
	v_mfma_f32_32x32x8f16 a[208:223], v[64:65], v[88:89], a[208:223]
	v_mfma_f32_32x32x8f16 a[176:191], v[64:65], v[96:97], a[176:191]
	v_add_u32_e32 v64, s34, v13
	v_mfma_f32_32x32x8f16 a[160:175], v[62:63], v[70:71], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[62:63], v[78:79], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[62:63], v[86:87], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[62:63], v[94:95], a[112:127]
	ds_read2_b64 v[60:63], v12 offset1:1
	v_mfma_f32_32x32x8f16 a[192:207], v[66:67], v[74:75], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[66:67], v[82:83], a[224:239]
	v_mfma_f32_32x32x8f16 a[208:223], v[66:67], v[90:91], a[208:223]
	v_mfma_f32_32x32x8f16 a[176:191], v[66:67], v[98:99], a[176:191]
	ds_read2_b64 v[64:67], v64 offset0:32 offset1:33
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[96:111], v[60:61], v[68:69], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[60:61], v[76:77], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[60:61], v[84:85], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[60:61], v[92:93], a[48:63]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[160:175], v[64:65], v[72:73], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[64:65], v[80:81], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[64:65], v[88:89], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[64:65], v[96:97], a[112:127]
	v_add_u32_e32 v64, s34, v12
	v_mfma_f32_32x32x8f16 a[96:111], v[62:63], v[70:71], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[62:63], v[78:79], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[62:63], v[86:87], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[62:63], v[94:95], a[48:63]
	ds_read2_b64 v[110:113], v9 offset1:1
	v_mfma_f32_32x32x8f16 a[160:175], v[66:67], v[74:75], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[66:67], v[82:83], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[66:67], v[90:91], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[66:67], v[98:99], a[112:127]
	ds_read2_b64 v[106:109], v64 offset0:32 offset1:33
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[32:47], v[110:111], v[68:69], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[110:111], v[76:77], a[16:31]
	v_mfma_f32_32x32x8f16 a[0:15], v[110:111], v[84:85], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[110:111], v[92:93], a[240:255]
	v_add3_u32 v60, v15, v101, s43
	v_lshrrev_b32_e32 v60, s11, v60
	v_mul_lo_u32 v61, s33, v60
	v_sub_u32_e32 v22, v60, v22
	v_lshl_add_u32 v22, v22, 3, -3
	v_mul_lo_u32 v22, v22, s2
	v_sub_u32_e32 v21, v61, v21
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[96:111], v[106:107], v[72:73], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[106:107], v[80:81], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[106:107], v[88:89], a[64:79]
	;v_mfma_f32_32x32x8f16 a[48:63], v[106:107], v[96:97], a[48:63]
	v_add_u32_e32 v64, s34, v9
	ds_read2_b64 v[102:105], v64 offset0:32 offset1:33;+
	v_mfma_f32_32x32x8f16 a[32:47], v[112:113], v[70:71], a[32:47]
	;v_mfma_f32_32x32x8f16 a[16:31], v[112:113], v[78:79], a[16:31]
	;v_mfma_f32_32x32x8f16 a[0:15], v[112:113], v[86:87], a[0:15]
	;v_mfma_f32_32x32x8f16 a[240:255], v[112:113], v[94:95], a[240:255]
	v_add_u32_e32 v62, s43, v15
	v_add_u32_e32 v21, v62, v21
	v_mul_lo_u32 v21, v21, s3
	v_add_u32_e32 v63, v62, v61
	v_add3_u32 v23, v22, v23, v21
	v_add3_u32 v21, v14, v100, s43
	v_lshrrev_b32_e32 v21, s41, v21
	v_mul_lo_u32 v61, s30, v21
	v_mul_lo_u32 v22, v21, s37
	v_sub_u32_e32 v20, v61, v20

	v_mfma_f32_32x32x8f16 a[16:31], v[112:113], v[78:79], a[16:31];+

	v_mul_hi_u32 v61, v21, s38
	;v_mfma_f32_32x32x8f16 a[96:111], v[66:67], v[74:75], a[96:111]
	v_add3_u32 v20, v14, s43, v20
	v_mul_lo_u32 v20, v20, s35
	v_add_u32_e32 v61, v21, v61
	v_lshrrev_b32_e32 v61, s40, v61
	v_mul_lo_u32 v62, v61, s36
	v_sub_u32_e32 v18, v61, v18
	v_lshl_add_u32 v18, v18, 3, -3
	v_add_u32_e32 v2, v20, v2
	v_sub_u32_e32 v62, v21, v62
	v_sub_u32_e32 v19, v62, v19

	v_mfma_f32_32x32x8f16 a[0:15], v[112:113], v[86:87], a[0:15];+

	v_mul_lo_u32 v19, v19, s31
	v_mul_lo_u32 v20, v20, s22
	v_mul_lo_u32 v18, v18, s20
	v_sub_u32_e32 v22, v1, v22
	v_add_u32_e32 v3, v19, v3
	;v_mfma_f32_32x32x8f16 a[80:95], v[66:67], v[82:83], a[80:95]
	v_mul_lo_u32 v19, v19, s21
	v_add_u32_e32 v20, v20, v24
	s_add_i32 s43, s43, 4
	s_cmp_lt_i32 s43, s29
	v_add3_u32 v24, v20, v18, v19
	;v_mfma_f32_32x32x8f16 a[64:79], v[66:67], v[90:91], a[64:79]
	;v_mfma_f32_32x32x8f16 a[48:63], v[66:67], v[98:99], a[48:63]
	;ds_read2_b64 v[102:105], v64 offset0:32 offset1:33
	;;#ASMSTART

	v_mfma_f32_32x32x8f16 a[240:255], v[112:113], v[94:95], a[240:255];+

	    s_waitcnt lgkmcnt(0) 
     s_barrier     

	v_mfma_f32_32x32x8f16 a[48:63], v[106:107], v[96:97], a[48:63];+

	;;#ASMEND
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	
             v_pack_b32_f16 v18, v25, v29 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v20, v25, v29, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v19, v36, v40 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v21, v36, v40, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v25, v26, v30 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v29, v26, v30, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v26, v37, v41 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v30, v37, v41, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v36, v27, v31 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v40, v27, v31, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v37, v38, v42 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v41, v38, v42, op_sel:[1, 1] 
             
	;;#ASMEND

	v_mfma_f32_32x32x8f16 a[96:111], v[108:109], v[74:75], a[96:111];+

	;;#ASMSTART
	
             v_pack_b32_f16 v27, v28, v32 
             
	;;#ASMEND
	;s_waitcnt lgkmcnt(0)
	;v_mfma_f32_32x32x8f16 a[32:47], v[102:103], v[72:73], a[32:47]
	;;#ASMSTART
	
             v_pack_b32_f16 v31, v28, v32, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v28, v39, v43 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v32, v39, v43, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v4, v[18:19], v[20:21] offset1:2
	v_mfma_f32_32x32x8f16 a[80:95], v[108:109], v[82:83], a[80:95];+
	ds_write2_b64 v4, v[25:26], v[29:30] offset0:4 offset1:6
	v_mfma_f32_32x32x8f16 a[64:79], v[108:109], v[90:91], a[64:79];+
	ds_write2_b64 v4, v[36:37], v[40:41] offset0:8 offset1:10
	v_mfma_f32_32x32x8f16 a[48:63], v[108:109], v[98:99], a[48:63];+
	ds_write2_b64 v4, v[27:28], v[31:32] offset0:12 offset1:14

	;s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8f16 a[32:47], v[102:103], v[72:73], a[32:47];+

	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v18, v44, v48 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v20, v44, v48, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v19, v52, v56 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v21, v52, v56, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v25, v45, v49 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v27, v45, v49, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v26, v53, v57 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v28, v53, v57, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v29, v46, v50 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v31, v46, v50, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v30, v54, v58 
             
	;;#ASMEND
	v_mfma_f32_32x32x8f16 a[16:31], v[102:103], v[80:81], a[16:31]
	;;#ASMSTART
	
             v_pack_b32_f16 v32, v54, v58, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v36, v47, v51 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v38, v47, v51, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v37, v55, v59 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v39, v55, v59, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v6, v[18:19], v[20:21] offset1:2
	v_mfma_f32_32x32x8f16 a[0:15], v[102:103], v[88:89], a[0:15];+
	ds_write2_b64 v6, v[25:26], v[27:28] offset0:4 offset1:6
	v_mfma_f32_32x32x8f16 a[240:255], v[102:103], v[96:97], a[240:255];+
	ds_write2_b64 v6, v[29:30], v[31:32] offset0:8 offset1:10
	v_mfma_f32_32x32x8f16 a[32:47], v[104:105], v[74:75], a[32:47];+
	ds_write2_b64 v6, v[36:37], v[38:39] offset0:12 offset1:14
	v_mov_b32_e32 v18, v61
	v_mov_b32_e32 v19, v62
	v_mov_b32_e32 v20, v22
	v_mov_b32_e32 v21, v63
	v_mov_b32_e32 v22, v60
	;v_mfma_f32_32x32x8f16 a[0:15], v[64:65], v[88:89], a[0:15]
	;v_mfma_f32_32x32x8f16 a[240:255], v[64:65], v[96:97], a[240:255]
	;v_mfma_f32_32x32x8f16 a[32:47], v[66:67], v[74:75], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[104:105], v[82:83], a[16:31]
	v_mfma_f32_32x32x8f16 a[0:15], v[104:105], v[90:91], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[104:105], v[98:99], a[240:255]
	s_cbranch_scc1 BB0_1
; %bb.2:                                ; %_ZZN2ck23Merge_v2_magic_divisionINS_5TupleIJNS_17integral_constantIiLi4EEENS2_IiLi2EEEiiiEEEEC1ERKS5_ENKUlT_E_clIS4_EEDaS9_.exit.i.i.i.i.i.i.i.i
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[1:4], v7 offset1:1
	s_movk_i32 s0, 0x1000
	v_add_u32_e32 v6, s0, v7
	ds_read2_b64 v[36:39], v6 offset0:32 offset1:33
	ds_read2_b64 v[14:17], v5 offset1:1
	ds_read2_b64 v[18:21], v8 offset1:1
	ds_read2_b64 v[22:25], v10 offset1:1
	ds_read2_b64 v[26:29], v11 offset1:1
	v_add_u32_e32 v5, s0, v5
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8f16 a[192:207], v[1:2], v[14:15], a[192:207]
	ds_read2_b64 v[40:43], v5 offset0:32 offset1:33
	v_add_u32_e32 v5, s0, v8
	ds_read2_b64 v[5:8], v5 offset0:32 offset1:33
	v_add_u32_e32 v10, s0, v10
	ds_read2_b64 v[44:47], v10 offset0:32 offset1:33
	v_add_u32_e32 v10, s0, v11
	ds_read2_b64 v[48:51], v10 offset0:32 offset1:33
	v_add_u32_e32 v10, s0, v13
	ds_read2_b64 v[52:55], v10 offset0:32 offset1:33
	v_add_u32_e32 v10, s0, v12
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8f16 a[224:239], v[1:2], v[18:19], a[224:239]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8f16 a[208:223], v[1:2], v[22:23], a[208:223]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8f16 a[176:191], v[1:2], v[26:27], a[176:191]
	v_mfma_f32_32x32x8f16 a[192:207], v[3:4], v[16:17], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[3:4], v[20:21], a[224:239]
	v_mfma_f32_32x32x8f16 a[208:223], v[3:4], v[24:25], a[208:223]
	v_mfma_f32_32x32x8f16 a[176:191], v[3:4], v[28:29], a[176:191]
	ds_read2_b64 v[1:4], v13 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[160:175], v[1:2], v[14:15], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[1:2], v[18:19], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[1:2], v[22:23], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[1:2], v[26:27], a[112:127]
	v_mfma_f32_32x32x8f16 a[160:175], v[3:4], v[16:17], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[3:4], v[20:21], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[3:4], v[24:25], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[3:4], v[28:29], a[112:127]
	ds_read2_b64 v[1:4], v12 offset1:1
	ds_read2_b64 v[10:13], v10 offset0:32 offset1:33
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[96:111], v[1:2], v[14:15], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[1:2], v[18:19], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[1:2], v[22:23], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[26:27], a[48:63]
	v_mfma_f32_32x32x8f16 a[96:111], v[3:4], v[16:17], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[3:4], v[20:21], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[3:4], v[24:25], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[28:29], a[48:63]
	ds_read2_b64 v[1:4], v9 offset1:1
	v_add_u32_e32 v9, s0, v9
	ds_read2_b64 v[56:59], v9 offset0:32 offset1:33
	s_movk_i32 s0, 0x80
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[18:19], a[16:31]
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[20:21], a[16:31]
	v_mfma_f32_32x32x8f16 a[32:47], v[1:2], v[14:15], a[32:47]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[16:31], v[56:57], v[5:6], a[16:31]
	v_mfma_f32_32x32x8f16 a[0:15], v[1:2], v[22:23], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[1:2], v[26:27], a[240:255]
	v_mfma_f32_32x32x8f16 a[192:207], v[36:37], v[40:41], a[192:207]
	v_mfma_f32_32x32x8f16 a[32:47], v[3:4], v[16:17], a[32:47]
	v_mfma_f32_32x32x8f16 a[0:15], v[3:4], v[24:25], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[3:4], v[28:29], a[240:255]
	v_mfma_f32_32x32x8f16 a[16:31], v[58:59], v[7:8], a[16:31]
	v_mfma_f32_32x32x8f16 a[192:207], v[38:39], v[42:43], a[192:207]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v3, a16              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:4 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a17              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:8 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a18              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:12 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a19              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[224:239], v[36:37], v[5:6], a[224:239]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:16 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a20              ;  Reload Reuse
	v_accvgpr_read_b32 v17, a192
	v_accvgpr_read_b32 v18, a193
	buffer_store_dword v3, off, s[64:67], 0 offset:20 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a21              ;  Reload Reuse
	v_accvgpr_read_b32 v19, a194
	v_accvgpr_read_b32 v20, a195
	buffer_store_dword v3, off, s[64:67], 0 offset:24 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a22              ;  Reload Reuse
	v_accvgpr_read_b32 v21, a196
	v_accvgpr_read_b32 v22, a197
	buffer_store_dword v3, off, s[64:67], 0 offset:28 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a23              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[208:223], v[36:37], v[44:45], a[208:223]
	v_accvgpr_read_b32 v23, a198
	buffer_store_dword v3, off, s[64:67], 0 offset:32 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a24              ;  Reload Reuse
	v_accvgpr_read_b32 v24, a199
	v_accvgpr_read_b32 v25, a200
	buffer_store_dword v3, off, s[64:67], 0 offset:36 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a25              ;  Reload Reuse
	v_accvgpr_read_b32 v26, a201
	v_accvgpr_read_b32 v27, a202
	buffer_store_dword v3, off, s[64:67], 0 offset:40 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a26              ;  Reload Reuse
	v_accvgpr_read_b32 v28, a203
	v_accvgpr_read_b32 v29, a204
	buffer_store_dword v3, off, s[64:67], 0 offset:44 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a27              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[176:191], v[36:37], v[48:49], a[176:191]
	v_accvgpr_read_b32 v30, a205
	buffer_store_dword v3, off, s[64:67], 0 offset:48 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a28              ;  Reload Reuse
	v_accvgpr_read_b32 v31, a206
	v_accvgpr_read_b32 v32, a207
	buffer_store_dword v3, off, s[64:67], 0 offset:52 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a29              ;  Reload Reuse
	v_mul_i32_i24_e32 v36, 0xffffffe0, v33
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:56 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a30              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:60 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a31              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[160:175], v[52:53], v[40:41], a[160:175]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:64 ; 4-byte Folded Spill
	v_mfma_f32_32x32x8f16 a[144:159], v[52:53], v[5:6], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[52:53], v[44:45], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[52:53], v[48:49], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[10:11], v[40:41], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[10:11], v[5:6], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[10:11], v[44:45], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[10:11], v[48:49], a[48:63]
	v_mfma_f32_32x32x8f16 a[32:47], v[56:57], v[40:41], a[32:47]
	v_mfma_f32_32x32x8f16 a[0:15], v[56:57], v[44:45], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[56:57], v[48:49], a[240:255]
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mfma_f32_32x32x8f16 a[224:239], v[38:39], v[7:8], a[224:239]
	v_mfma_f32_32x32x8f16 a[176:191], v[38:39], v[50:51], a[176:191]
	v_mfma_f32_32x32x8f16 a[144:159], v[54:55], v[7:8], a[144:159]
	v_mfma_f32_32x32x8f16 a[112:127], v[54:55], v[50:51], a[112:127]
	v_mfma_f32_32x32x8f16 a[80:95], v[12:13], v[7:8], a[80:95]
	v_mfma_f32_32x32x8f16 a[48:63], v[12:13], v[50:51], a[48:63]
	v_mfma_f32_32x32x8f16 a[16:31], v[58:59], v[50:51], a[240:255]
	v_mfma_f32_32x32x8f16 a[192:207], v[54:55], v[42:43], a[160:175]
	v_mfma_f32_32x32x8f16 a[240:255], v[12:13], v[42:43], a[96:111]
	v_mfma_f32_32x32x8f16 a[96:111], v[58:59], v[42:43], a[32:47]
	v_mfma_f32_32x32x8f16 a[208:223], v[38:39], v[46:47], a[208:223]
	v_mfma_f32_32x32x8f16 a[64:79], v[12:13], v[46:47], a[64:79]
	v_mfma_f32_32x32x8f16 a[160:175], v[54:55], v[46:47], a[128:143]
	v_mfma_f32_32x32x8f16 a[32:47], v[58:59], v[46:47], a[0:15]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_4
; %bb.3:
	v_lshrrev_b32_e32 v1, 2, v0
	v_mul_i32_i24_e32 v2, -4, v1
	v_add_u32_e32 v1, v36, v1
	v_lshlrev_b32_e32 v3, 1, v1
	v_add_u32_e32 v4, s6, v33
	v_lshl_add_u32 v3, v4, 8, v3
	v_mul_lo_u32 v3, v3, s7
	v_add_lshl_u32 v2, v2, v0, 4
	v_lshlrev_b32_e32 v4, 12, v33
	v_lshlrev_b32_e32 v1, 7, v1
	v_add3_u32 v49, v2, v4, v1
	v_add3_u32 v48, s5, v2, v3
BB0_4:                                  ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EEC2ERSO_RKNSA_IJiiiiEEES15_S1A_RKS3_.exit.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_or_b32 v0, v0, 4, v34
	v_lshlrev_b32_e32 v33, 5, v33
	v_lshrrev_b32_e32 v34, 6, v35
	v_add3_u32 v0, v0, v36, v33
	v_sub_u32_e32 v0, v0, v34
	v_lshlrev_b32_e32 v0, 6, v0
	v_cvt_f16_f32_e32 v17, v17
	v_add_lshl_u32 v50, v0, v35, 1
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v18, v19
	v_cvt_f16_f32_e32 v19, v20
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v17
	ds_write_b16 v50, v0 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v17, v23
	v_cvt_f16_f32_e32 v18, v22
	v_cvt_f16_f32_e32 v19, v21
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v17, v26
	v_cvt_f16_f32_e32 v18, v27
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v17, v31
	v_accvgpr_read_b32 v1, a224
	v_cvt_f16_f32_e32 v18, v30
	v_accvgpr_read_b32 v2, a225
	v_accvgpr_read_b32 v3, a226
	v_accvgpr_read_b32 v4, a227
	v_accvgpr_read_b32 v5, a228
	v_accvgpr_read_b32 v6, a229
	v_accvgpr_read_b32 v7, a230
	v_accvgpr_read_b32 v8, a231
	v_accvgpr_read_b32 v9, a232
	v_accvgpr_read_b32 v10, a233
	v_accvgpr_read_b32 v11, a234
	v_accvgpr_read_b32 v12, a235
	v_accvgpr_read_b32 v13, a236
	v_accvgpr_read_b32 v14, a237
	v_accvgpr_read_b32 v15, a238
	v_accvgpr_read_b32 v16, a239
	v_cvt_f16_f32_e32 v19, v29
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_6
; %bb.5:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i.i.i.i
	v_lshlrev_b32_e32 v0, 1, v49
	ds_read_b128 v[17:20], v0
	ds_read_b128 v[21:24], v0 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[17:20], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v17, 8, v48
	v_lshlrev_b32_e32 v18, 1, v17
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[21:24], v18, s[8:11], 0 offen
	v_add_lshl_u32 v25, v17, s7, 1
	ds_read_b128 v[17:20], v0 offset:144
	ds_read_b128 v[21:24], v0 offset:128
	v_add_lshl_u32 v0, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[17:20], v25, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[21:24], v0, s[8:11], 0 offen
BB0_6:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v1, v2
	v_cvt_f16_f32_e32 v2, v3
	v_cvt_f16_f32_e32 v3, v4
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v7
	v_cvt_f16_f32_e32 v2, v6
	v_cvt_f16_f32_e32 v3, v5
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v9
	v_cvt_f16_f32_e32 v1, v10
	v_cvt_f16_f32_e32 v2, v11
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v15
	v_accvgpr_read_b32 v32, a208
	v_cvt_f16_f32_e32 v2, v14
	v_accvgpr_read_b32 v33, a209
	v_accvgpr_read_b32 v34, a210
	v_accvgpr_read_b32 v35, a211
	v_accvgpr_read_b32 v36, a212
	v_accvgpr_read_b32 v37, a213
	v_accvgpr_read_b32 v38, a214
	v_accvgpr_read_b32 v39, a215
	v_accvgpr_read_b32 v40, a216
	v_accvgpr_read_b32 v41, a217
	v_accvgpr_read_b32 v42, a218
	v_accvgpr_read_b32 v43, a219
	v_accvgpr_read_b32 v44, a220
	v_accvgpr_read_b32 v45, a221
	v_accvgpr_read_b32 v46, a222
	v_accvgpr_read_b32 v47, a223
	v_cvt_f16_f32_e32 v3, v13
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_8
; %bb.7:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i81.i.i.i.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_8:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_106.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v1, v33
	v_cvt_f16_f32_e32 v2, v34
	v_cvt_f16_f32_e32 v3, v35
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v39
	v_cvt_f16_f32_e32 v1, v38
	v_cvt_f16_f32_e32 v2, v37
	v_cvt_f16_f32_e32 v3, v36
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v40
	v_cvt_f16_f32_e32 v1, v41
	v_cvt_f16_f32_e32 v2, v42
	v_cvt_f16_f32_e32 v3, v43
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v47
	v_cvt_f16_f32_e32 v1, v46
	v_accvgpr_read_b32 v16, a176
	v_cvt_f16_f32_e32 v2, v45
	v_accvgpr_read_b32 v17, a177
	v_accvgpr_read_b32 v18, a178
	v_accvgpr_read_b32 v19, a179
	v_accvgpr_read_b32 v20, a180
	v_accvgpr_read_b32 v21, a181
	v_accvgpr_read_b32 v22, a182
	v_accvgpr_read_b32 v23, a183
	v_accvgpr_read_b32 v24, a184
	v_accvgpr_read_b32 v25, a185
	v_accvgpr_read_b32 v26, a186
	v_accvgpr_read_b32 v27, a187
	v_accvgpr_read_b32 v28, a188
	v_accvgpr_read_b32 v29, a189
	v_accvgpr_read_b32 v30, a190
	v_accvgpr_read_b32 v31, a191
	v_cvt_f16_f32_e32 v3, v44
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_10
; %bb.9:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i187.i.i.i.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_10:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_212.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a112
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a113
	v_accvgpr_read_b32 v2, a114
	v_accvgpr_read_b32 v3, a115
	v_accvgpr_read_b32 v4, a116
	v_accvgpr_read_b32 v5, a117
	v_accvgpr_read_b32 v6, a118
	v_accvgpr_read_b32 v7, a119
	v_accvgpr_read_b32 v8, a120
	v_accvgpr_read_b32 v9, a121
	v_accvgpr_read_b32 v10, a122
	v_accvgpr_read_b32 v11, a123
	v_accvgpr_read_b32 v12, a124
	v_accvgpr_read_b32 v13, a125
	v_accvgpr_read_b32 v14, a126
	v_accvgpr_read_b32 v15, a127
	v_cvt_f16_f32_e32 v19, v28
	s_mul_i32 s2, s7, 63
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_12
; %bb.11:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
	v_add_lshl_u32 v25, v16, s7, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, s7, v48
	v_lshlrev_b32_e32 v17, 1, v16
	v_add_u32_e32 v48, s2, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
BB0_12:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a160
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a161
	v_accvgpr_read_b32 v18, a162
	v_accvgpr_read_b32 v19, a163
	v_accvgpr_read_b32 v20, a164
	v_accvgpr_read_b32 v21, a165
	v_accvgpr_read_b32 v22, a166
	v_accvgpr_read_b32 v23, a167
	v_accvgpr_read_b32 v24, a168
	v_accvgpr_read_b32 v25, a169
	v_accvgpr_read_b32 v26, a170
	v_accvgpr_read_b32 v27, a171
	v_accvgpr_read_b32 v28, a172
	v_accvgpr_read_b32 v29, a173
	v_accvgpr_read_b32 v30, a174
	v_accvgpr_read_b32 v31, a175
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_14
; %bb.13:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i92.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_14:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i140.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a144
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a145
	v_accvgpr_read_b32 v2, a146
	v_accvgpr_read_b32 v3, a147
	v_accvgpr_read_b32 v4, a148
	v_accvgpr_read_b32 v5, a149
	v_accvgpr_read_b32 v6, a150
	v_accvgpr_read_b32 v7, a151
	v_accvgpr_read_b32 v8, a152
	v_accvgpr_read_b32 v9, a153
	v_accvgpr_read_b32 v10, a154
	v_accvgpr_read_b32 v11, a155
	v_accvgpr_read_b32 v12, a156
	v_accvgpr_read_b32 v13, a157
	v_accvgpr_read_b32 v14, a158
	v_accvgpr_read_b32 v15, a159
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_16
; %bb.15:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i81.i.i.i192.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
	v_add_lshl_u32 v25, v16, s7, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v16, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v16, s[8:11], 0 offen
BB0_16:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_106.i.i.i240.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a192
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a193
	v_accvgpr_read_b32 v18, a194
	v_accvgpr_read_b32 v19, a195
	v_accvgpr_read_b32 v20, a196
	v_accvgpr_read_b32 v21, a197
	v_accvgpr_read_b32 v22, a198
	v_accvgpr_read_b32 v23, a199
	v_accvgpr_read_b32 v24, a200
	v_accvgpr_read_b32 v25, a201
	v_accvgpr_read_b32 v26, a202
	v_accvgpr_read_b32 v27, a203
	v_accvgpr_read_b32 v28, a204
	v_accvgpr_read_b32 v29, a205
	v_accvgpr_read_b32 v30, a206
	v_accvgpr_read_b32 v31, a207
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_18
; %bb.17:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i187.i.i.i292.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_18:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_212.i.i.i340.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a240
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a241
	v_accvgpr_read_b32 v2, a242
	v_accvgpr_read_b32 v3, a243
	v_accvgpr_read_b32 v4, a244
	v_accvgpr_read_b32 v5, a245
	v_accvgpr_read_b32 v6, a246
	v_accvgpr_read_b32 v7, a247
	v_accvgpr_read_b32 v8, a248
	v_accvgpr_read_b32 v9, a249
	v_accvgpr_read_b32 v10, a250
	v_accvgpr_read_b32 v11, a251
	v_accvgpr_read_b32 v12, a252
	v_accvgpr_read_b32 v13, a253
	v_accvgpr_read_b32 v14, a254
	v_accvgpr_read_b32 v15, a255
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_20
; %bb.19:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i380.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
	v_add_lshl_u32 v25, v16, s7, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, s7, v48
	v_lshlrev_b32_e32 v17, 1, v16
	v_add_u32_e32 v48, s2, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
BB0_20:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I405.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a80
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a81
	v_accvgpr_read_b32 v18, a82
	v_accvgpr_read_b32 v19, a83
	v_accvgpr_read_b32 v20, a84
	v_accvgpr_read_b32 v21, a85
	v_accvgpr_read_b32 v22, a86
	v_accvgpr_read_b32 v23, a87
	v_accvgpr_read_b32 v24, a88
	v_accvgpr_read_b32 v25, a89
	v_accvgpr_read_b32 v26, a90
	v_accvgpr_read_b32 v27, a91
	v_accvgpr_read_b32 v28, a92
	v_accvgpr_read_b32 v29, a93
	v_accvgpr_read_b32 v30, a94
	v_accvgpr_read_b32 v31, a95
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_22
; %bb.21:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i497.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_22:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i545.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a64
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a65
	v_accvgpr_read_b32 v2, a66
	v_accvgpr_read_b32 v3, a67
	v_accvgpr_read_b32 v4, a68
	v_accvgpr_read_b32 v5, a69
	v_accvgpr_read_b32 v6, a70
	v_accvgpr_read_b32 v7, a71
	v_accvgpr_read_b32 v8, a72
	v_accvgpr_read_b32 v9, a73
	v_accvgpr_read_b32 v10, a74
	v_accvgpr_read_b32 v11, a75
	v_accvgpr_read_b32 v12, a76
	v_accvgpr_read_b32 v13, a77
	v_accvgpr_read_b32 v14, a78
	v_accvgpr_read_b32 v15, a79
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_24
; %bb.23:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i81.i.i.i597.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
	v_add_lshl_u32 v25, v16, s7, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v16, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v16, s[8:11], 0 offen
BB0_24:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_106.i.i.i645.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a48
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a49
	v_accvgpr_read_b32 v18, a50
	v_accvgpr_read_b32 v19, a51
	v_accvgpr_read_b32 v20, a52
	v_accvgpr_read_b32 v21, a53
	v_accvgpr_read_b32 v22, a54
	v_accvgpr_read_b32 v23, a55
	v_accvgpr_read_b32 v24, a56
	v_accvgpr_read_b32 v25, a57
	v_accvgpr_read_b32 v26, a58
	v_accvgpr_read_b32 v27, a59
	v_accvgpr_read_b32 v28, a60
	v_accvgpr_read_b32 v29, a61
	v_accvgpr_read_b32 v30, a62
	v_accvgpr_read_b32 v31, a63
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_26
; %bb.25:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i187.i.i.i697.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_26:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_212.i.i.i745.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a16
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a17
	v_accvgpr_read_b32 v2, a18
	v_accvgpr_read_b32 v3, a19
	v_accvgpr_read_b32 v4, a20
	v_accvgpr_read_b32 v5, a21
	v_accvgpr_read_b32 v6, a22
	v_accvgpr_read_b32 v7, a23
	v_accvgpr_read_b32 v8, a24
	v_accvgpr_read_b32 v9, a25
	v_accvgpr_read_b32 v10, a26
	v_accvgpr_read_b32 v11, a27
	v_accvgpr_read_b32 v12, a28
	v_accvgpr_read_b32 v13, a29
	v_accvgpr_read_b32 v14, a30
	v_accvgpr_read_b32 v15, a31
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_28
; %bb.27:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i785.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
	v_add_lshl_u32 v25, v16, s7, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, s7, v48
	v_lshlrev_b32_e32 v17, 1, v16
	v_add_u32_e32 v48, s2, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
BB0_28:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I810.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a32
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a33
	v_accvgpr_read_b32 v18, a34
	v_accvgpr_read_b32 v19, a35
	v_accvgpr_read_b32 v20, a36
	v_accvgpr_read_b32 v21, a37
	v_accvgpr_read_b32 v22, a38
	v_accvgpr_read_b32 v23, a39
	v_accvgpr_read_b32 v24, a40
	v_accvgpr_read_b32 v25, a41
	v_accvgpr_read_b32 v26, a42
	v_accvgpr_read_b32 v27, a43
	v_accvgpr_read_b32 v28, a44
	v_accvgpr_read_b32 v29, a45
	v_accvgpr_read_b32 v30, a46
	v_accvgpr_read_b32 v31, a47
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_30
; %bb.29:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i902.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_30:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i950.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v32, off, s[64:67], 0 offset:4 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:64 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_cvt_f16_f32_e32 v18, v29
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v32            ;  Reload Reuse
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v15, a15
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_32
; %bb.31:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i81.i.i.i1002.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
	v_add_lshl_u32 v25, v16, s7, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v16, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v16, s[8:11], 0 offen
BB0_32:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_106.i.i.i1050.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a96
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a97
	v_accvgpr_read_b32 v18, a98
	v_accvgpr_read_b32 v19, a99
	v_accvgpr_read_b32 v20, a100
	v_accvgpr_read_b32 v21, a101
	v_accvgpr_read_b32 v22, a102
	v_accvgpr_read_b32 v23, a103
	v_accvgpr_read_b32 v24, a104
	v_accvgpr_read_b32 v25, a105
	v_accvgpr_read_b32 v26, a106
	v_accvgpr_read_b32 v27, a107
	v_accvgpr_read_b32 v28, a108
	v_accvgpr_read_b32 v29, a109
	v_accvgpr_read_b32 v30, a110
	v_accvgpr_read_b32 v31, a111
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_34
; %bb.33:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i187.i.i.i1102.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_34:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_212.i.i.i1150.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v17
	v_cvt_f16_f32_e32 v2, v18
	v_cvt_f16_f32_e32 v3, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v23
	v_cvt_f16_f32_e32 v1, v22
	v_cvt_f16_f32_e32 v2, v21
	v_cvt_f16_f32_e32 v3, v20
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v25
	v_cvt_f16_f32_e32 v2, v26
	v_cvt_f16_f32_e32 v3, v27
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v31
	v_cvt_f16_f32_e32 v1, v30
	v_cvt_f16_f32_e32 v2, v29
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_36
; %bb.35:
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_36:                                 ; %_ZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_IJ
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
		.amdhsa_group_segment_fixed_size 34816
		.amdhsa_private_segment_fixed_size 68
		.amdhsa_kernarg_size 600
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 68
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_, .Lfunc_end0-_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 15084
; NumSgprs: 70
; NumVgprs: 102
; NumAgprs: 256
; TotalNumVgprs: 256
; ScratchSize: 68
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 34816 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 70
; NumVGPRsForWavesPerEU: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_ ; -- Begin function _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
	.globl	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
	.p2align	8
	.type	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_,@function
_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_: ; @_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
; %bb.0:
	s_mov_b64 s[66:67], s[2:3]
	s_mov_b64 s[64:65], s[0:1]
	s_add_u32 s64, s64, s7
	s_load_dwordx2 s[12:13], s[4:5], 0x0
	s_load_dwordx2 s[16:17], s[4:5], 0x8
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	s_load_dwordx2 s[2:3], s[4:5], 0x24
	s_load_dword s33, s[4:5], 0x48
	s_load_dword s10, s[4:5], 0x50
	s_load_dword s11, s[4:5], 0x58
	s_load_dwordx2 s[42:43], s[4:5], 0x6c
	s_load_dword s56, s[4:5], 0x84
	s_load_dwordx4 s[20:23], s[4:5], 0x98
	s_load_dwordx4 s[24:27], s[4:5], 0xac
	s_load_dwordx2 s[28:29], s[4:5], 0xbc
	s_load_dwordx2 s[30:31], s[4:5], 0xd4
	s_load_dwordx2 s[34:35], s[4:5], 0xe4
	s_load_dwordx2 s[36:37], s[4:5], 0x114
	s_load_dwordx2 s[38:39], s[4:5], 0x120
	s_load_dwordx2 s[40:41], s[4:5], 0x12c
	s_load_dwordx2 s[0:1], s[4:5], 0x13c
	s_load_dwordx2 s[14:15], s[4:5], 0x148
	s_load_dwordx2 s[18:19], s[4:5], 0x154
	s_load_dword s57, s[4:5], 0x16c
	s_load_dword s58, s[4:5], 0x180
	s_load_dword s7, s[4:5], 0x18c
	s_waitcnt lgkmcnt(0)
	s_load_dword s23, s[4:5], 0x1b0
	s_load_dword s59, s[4:5], 0x1c4
	s_load_dword s60, s[4:5], 0x1d4
	s_load_dwordx4 s[44:47], s[4:5], 0x1e0
	s_load_dwordx4 s[48:51], s[4:5], 0x1f4
	s_load_dwordx4 s[52:55], s[4:5], 0x208
	s_addc_u32 s65, s65, 0
	v_lshrrev_b32_e32 v1, 5, v0
	v_lshrrev_b32_e32 v33, 7, v0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s4, s51, s6
	s_add_i32 s4, s6, s4
	s_lshr_b32 s4, s4, s55
	s_mul_i32 s5, s4, s47
	s_sub_i32 s5, s6, s5
	s_mul_hi_u32 s6, s4, s50
	s_add_i32 s6, s4, s6
	s_lshr_b32 s6, s6, s54
	s_mul_i32 s46, s6, s46
	s_sub_i32 s4, s4, s46
	s_mul_hi_u32 s46, s6, s49
	s_add_i32 s46, s6, s46
	s_lshr_b32 s46, s46, s53
	s_mul_i32 s45, s46, s45
	s_sub_i32 s6, s6, s45
	s_mul_hi_u32 s45, s46, s48
	s_add_i32 s45, s46, s45
	s_lshr_b32 s45, s45, s52
	v_mad_i32_i24 v17, v33, -4, v1
	s_mul_i32 s43, s45, s43
	v_add_u32_e32 v50, s43, v17
	s_mul_i32 s44, s45, s44
	v_mul_hi_u32 v2, v50, s10
	s_sub_i32 s44, s46, s44
	s_mul_i32 s6, s6, s60
	s_add_i32 s5, s5, s6
	s_mul_i32 s6, s44, s59
	s_movk_i32 s44, 0xffe0
	s_add_i32 s6, s6, s4
	v_mad_i32_i24 v24, v1, s44, v0
	s_lshl_b32 s4, s6, 8
	s_lshl_b32 s5, s5, 8
	v_lshlrev_b32_e32 v1, 3, v24
	v_add_u32_e32 v2, v50, v2
	v_lshrrev_b32_e32 v18, s11, v2
	v_add_u32_e32 v3, s4, v1
	v_add_u32_e32 v1, s5, v1
	s_mul_i32 s45, s45, s57
	v_mul_lo_u32 v2, v18, s33
	v_mul_hi_u32 v4, v1, s15
	v_add_u32_e32 v51, s45, v17
	v_mul_hi_u32 v5, v51, s39
	v_sub_u32_e32 v20, v50, v2
	v_add_u32_e32 v2, v1, v4
	v_lshrrev_b32_e32 v2, s19, v2
	v_add_u32_e32 v5, v51, v5
	v_mul_hi_u32 v4, v2, s14
	v_lshrrev_b32_e32 v5, s41, v5
	v_mul_hi_u32 v7, v5, s38
	v_mul_lo_u32 v9, v5, s37
	v_add_u32_e32 v4, v2, v4
	v_lshrrev_b32_e32 v4, s18, v4
	v_add_u32_e32 v7, v5, v7
	v_mul_lo_u32 v8, v4, s0
	v_lshrrev_b32_e32 v52, s40, v7
	v_mul_lo_u32 v7, v52, s36
	v_sub_u32_e32 v53, v51, v9
	v_sub_u32_e32 v8, v2, v8
	v_mul_lo_u32 v4, v4, s30
	v_sub_u32_e32 v54, v5, v7
	v_mul_lo_u32 v5, v8, s34
	v_mul_lo_u32 v7, v53, s35
	v_mul_lo_u32 v8, v54, s31
	v_lshlrev_b32_e32 v19, 2, v33
	v_lshl_or_b32 v6, v18, 3, v19
	v_add_u32_e32 v55, v7, v5
	v_mul_lo_u32 v6, v6, s2
	v_mul_lo_u32 v10, v20, s3
	v_mul_lo_u32 v2, v2, s1
	v_add_u32_e32 v56, v8, v4
	v_subrev_u32_e32 v4, s28, v55
	v_lshl_or_b32 v9, v52, 3, v19
	v_subrev_u32_e32 v5, s25, v56
	v_mul_lo_u32 v4, v4, s22
	v_mul_lo_u32 v7, v9, s20
	v_mul_lo_u32 v5, v5, s21
	v_add3_u32 v3, v3, v6, v10
	v_sub_u32_e32 v1, v1, v2
	v_add_u32_e32 v1, v1, v4
	v_add_u32_e32 v9, s2, v3
	v_add3_u32 v21, v1, v7, v5
	v_lshlrev_b32_e32 v5, 1, v9
	v_add_u32_e32 v9, s2, v9
	s_lshl_b32 s14, s56, 1
	s_mov_b32 s15, 0x20000
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v22, 1, v9
	v_add_u32_e32 v23, s2, v9
	buffer_load_dwordx4 v[1:4], v1, s[12:15], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[5:8], v5, s[12:15], 0 offen
	v_lshlrev_b32_e32 v25, 1, v23
	buffer_load_dwordx4 v[9:12], v22, s[12:15], 0 offen
	buffer_load_dwordx4 v[13:16], v25, s[12:15], 0 offen
	s_sub_i32 s27, s27, s29
	v_cmp_le_i32_e32 vcc, s28, v55
	v_cmp_gt_i32_e64 s[0:1], s27, v55
	s_sub_i32 s24, s24, s26
	s_and_b64 s[44:45], vcc, s[0:1]
	v_cmp_le_i32_e32 vcc, s25, v56
	v_cmp_gt_i32_e64 s[0:1], s24, v56
	s_and_b64 s[0:1], vcc, s[0:1]
	s_brev_b32 s26, -2
	v_mov_b32_e32 v22, s26
	s_and_b64 s[0:1], s[44:45], s[0:1]
	v_cndmask_b32_e64 v22, v22, 0, s[0:1]
	v_lshl_add_u32 v34, v21, 1, v22
	v_add_u32_e32 v21, s20, v21
	s_lshl_b32 s18, s58, 1
	s_mov_b32 s19, s15
	v_lshl_add_u32 v35, v21, 1, v22
	v_add_u32_e32 v21, s20, v21
	buffer_load_dwordx4 v[25:28], v34, s[16:19], 0 offen
	buffer_load_dwordx4 v[29:32], v35, s[16:19], 0 offen
	v_lshl_add_u32 v34, v21, 1, v22
	v_add_u32_e32 v57, s20, v21
	v_lshl_add_u32 v21, v57, 1, v22
	buffer_load_dwordx4 v[36:39], v34, s[16:19], 0 offen
	buffer_load_dwordx4 v[40:43], v21, s[16:19], 0 offen
	v_lshlrev_b32_e32 v34, 5, v33
	s_movk_i32 s0, 0x880
	s_movk_i32 s1, 0x44
	s_movk_i32 s29, 0x80
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a204, 0
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	
             v_pack_b32_f16 v44, v1, v5 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v46, v1, v5, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v45, v9, v13 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v47, v9, v13, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v2, v6 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v2, v6, op_sel:[1, 1] 
             
	;;#ASMEND
	v_and_b32_e32 v1, 63, v0
	v_and_b32_e32 v2, 32, v0
	v_sub_u32_e32 v1, v1, v2
	v_add_u32_e32 v58, v1, v34
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v10, v14 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v10, v14, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v3, v7 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v48, v3, v7, op_sel:[1, 1] 
             
	;;#ASMEND
	v_ashrrev_i16_e32 v3, 15, v58
	v_lshrrev_b16_e32 v3, 13, v3
	v_add_u16_e32 v3, v58, v3
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v11, v15 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v49, v11, v15, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v4, v8 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v4, v8, op_sel:[1, 1] 
             
	;;#ASMEND
	v_ashrrev_i16_e32 v4, 3, v3
	v_and_b32_e32 v3, -8, v3
	v_lshrrev_b32_e32 v2, 4, v0
	v_sub_u16_e32 v3, v58, v3
	v_and_b32_e32 v2, 2, v2
	v_bfe_i32 v59, v4, 0, 16
	v_bfe_i32 v60, v3, 0, 16
	v_mul_u32_u24_e32 v2, s0, v2
	v_mul_i32_i24_e32 v3, s1, v59
	v_lshlrev_b32_e32 v4, 3, v60
	v_add3_u32 v61, v3, v2, v4
	v_lshrrev_b32_e32 v3, 6, v0
	v_mad_i32_i24 v3, v33, -2, v3
	v_lshl_add_u32 v35, v3, 5, v1
	v_ashrrev_i32_e32 v1, 31, v35
	v_lshrrev_b32_e32 v1, 29, v1
	v_add_u32_e32 v1, v35, v1
	v_add_u32_e32 v3, 4, v50
	v_ashrrev_i32_e32 v62, 3, v1
	v_mul_hi_u32 v4, v3, s10
	v_mul_lo_u32 v15, v62, s1
	v_and_b32_e32 v1, -8, v1
	v_sub_u32_e32 v63, v35, v1
	v_lshlrev_b32_e32 v1, 3, v63
	v_add_u32_e32 v4, v3, v4
	v_add3_u32 v64, v15, v2, v1
	v_add_u32_e32 v1, 4, v51
	v_lshrrev_b32_e32 v22, s11, v4
	v_mul_hi_u32 v2, v1, s39
	v_mul_lo_u32 v4, v22, s33
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v12, v16 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v12, v16, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v2, v1, v2
	v_sub_u32_e32 v21, v3, v4
	v_sub_u32_e32 v4, v22, v18
	v_lshrrev_b32_e32 v2, s41, v2
	v_sub_u32_e32 v3, v21, v20
	v_lshl_add_u32 v4, v4, 3, -3
	v_mul_hi_u32 v15, v2, s38
	v_mul_lo_u32 v4, v4, s2
	v_mul_lo_u32 v3, v3, s3
	v_mul_lo_u32 v16, v17, s0
	v_add_u32_e32 v15, v2, v15
	v_lshrrev_b32_e32 v18, s40, v15
	v_add3_u32 v23, v3, v4, v23
	v_mul_lo_u32 v3, v2, s37
	v_mul_lo_u32 v15, v18, s36
	v_or_b32_e32 v4, v16, v19
	s_movk_i32 s0, 0x4400
	v_sub_u32_e32 v20, v1, v3
	v_sub_u32_e32 v19, v2, v15
	v_sub_u32_e32 v2, v20, v53
	v_sub_u32_e32 v3, v19, v54
	v_mul_lo_u32 v15, v2, s35
	v_mul_lo_u32 v16, v3, s31
	v_sub_u32_e32 v2, v18, v52
	v_lshl_add_u32 v17, v2, 3, -3
	v_mul_lo_u32 v52, v24, s1
	v_add_u32_e32 v2, v15, v55
	v_mul_lo_u32 v17, v17, s20
	v_mul_lo_u32 v15, v15, s22
	v_add_u32_e32 v3, v16, v56
	v_mul_lo_u32 v16, v16, s21
	v_add_lshl_u32 v4, v4, v52, 1
	v_add_u32_e32 v15, v17, v15
	ds_write2_b64 v4, v[44:45], v[46:47] offset1:2
	ds_write2_b64 v4, v[5:6], v[9:10] offset0:4 offset1:6
	ds_write2_b64 v4, v[13:14], v[48:49] offset0:8 offset1:10
	ds_write2_b64 v4, v[7:8], v[11:12] offset0:12 offset1:14
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v25, v29 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v25, v29, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v36, v40 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v36, v40, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v5, 0x4000, v4
	v_add3_u32 v24, v15, v16, v57
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v26, v30 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v26, v30, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v37, v41 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v37, v41, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v15, v27, v31 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v25, v27, v31, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v16, v38, v42 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v26, v38, v42, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v27, v28, v32 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v29, v28, v32, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v28, v39, v43 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v30, v39, v43, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v5, v[7:8], v[9:10] offset0:128 offset1:130
	ds_write2_b64 v5, v[11:12], v[13:14] offset0:132 offset1:134
	ds_write2_b64 v5, v[15:16], v[25:26] offset0:136 offset1:138
	ds_write2_b64 v5, v[27:28], v[29:30] offset0:140 offset1:142
	v_add_u32_e32 v8, 64, v35
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshrrev_b32_e32 v9, 29, v9
	v_add_u32_e32 v9, v8, v9
	v_ashrrev_i32_e32 v10, 3, v9
	v_sub_u32_e32 v10, v10, v62
	v_add_u32_e32 v6, s0, v4
	v_lshl_add_u32 v5, v64, 1, s0
	s_mov_b32 s0, 0xffffff8
	v_mul_lo_u32 v10, v10, s1
	v_and_b32_e32 v9, s0, v9
	v_sub_u32_e32 v8, v8, v9
	v_sub_u32_e32 v8, v8, v63
	v_add_u32_e32 v9, s29, v35
	v_lshl_add_u32 v8, v8, 3, v10
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshrrev_b32_e32 v10, 29, v10
	v_add_u32_e32 v10, v9, v10
	v_ashrrev_i32_e32 v11, 3, v10
	v_sub_u32_e32 v11, v11, v62
	v_mul_lo_u32 v11, v11, s1
	v_and_b32_e32 v10, s0, v10
	v_sub_u32_e32 v9, v9, v10
	v_sub_u32_e32 v9, v9, v63
	v_lshl_add_u32 v9, v9, 3, v11
	s_movk_i32 s0, 0xc0
	v_lshl_add_u32 v10, v9, 1, v5
	v_add_u32_e32 v9, s0, v35
	v_ashrrev_i32_e32 v11, 31, v9
	v_lshrrev_b32_e32 v11, 29, v11
	v_add_u32_e32 v11, v9, v11
	v_ashrrev_i32_e32 v12, 3, v11
	v_sub_u32_e32 v12, v12, v62
	v_mul_lo_u32 v12, v12, s1
	v_and_b32_e32 v11, 0xffffff8, v11
	v_sub_u32_e32 v9, v9, v11
	v_add_u32_e32 v13, s29, v58
	v_sub_u32_e32 v9, v9, v63
	v_lshrrev_b32_e32 v13, 3, v13
	v_lshl_add_u32 v9, v9, 3, v12
	v_sub_u32_e32 v13, v13, v59
	v_lshl_add_u32 v11, v9, 1, v5
	v_add_u32_e32 v9, 64, v58
	v_mul_lo_u32 v14, v13, s1
	v_add_u32_e32 v13, s0, v58
	v_lshrrev_b32_e32 v9, 3, v9
	v_lshrrev_b32_e32 v13, 3, v13
	v_sub_u32_e32 v9, v9, v59
	v_sub_u32_e32 v13, v13, v59
	v_mul_lo_u32 v9, v9, s1
	v_mul_lo_u32 v15, v13, s1
	v_and_b32_e32 v12, 7, v58
	v_sub_u32_e32 v12, v12, v60
	v_lshl_add_u32 v16, v12, 3, v61
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a255, 0
	v_add_lshl_u32 v13, v16, v9, 1
	v_add_lshl_u32 v12, v16, v14, 1
	v_add_lshl_u32 v9, v16, v15, 1
	v_add_u32_e32 v14, 8, v51
	v_add_u32_e32 v15, 8, v50
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a240, 0
	s_mov_b32 s43, 0
	s_mov_b32 s4, s39
	v_lshlrev_b32_e32 v7, 1, v61
	v_lshl_add_u32 v8, v8, 1, v5
	s_add_i32 s29, s42, -4
	s_sub_i32 s30, 0, s37
	s_sub_i32 s33, 0, s33
	s_movk_i32 s34, 0x1000
	v_mov_b32_e32 v16, v15
	v_mov_b32_e32 v17, v14
BB1_1:                                  ; %_ZZN2ck22move_tensor_coordinateINS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS2_IJiiiEEELb0EEENS3_INS2_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESB_NS_23Merge_v2_magic_divisionINS2_IJiiEEEEESB_NSA_IS7_EENS3_ISD_Lb0EEESB_SF_EEENS2_IJNS_8SequenceIJLi0EEEENSI_IJLi1EEEENSI_IJLi2EEEENSI_IJLi3EEEENSI_IJLi4ELi6EEEENSI_IJLi7EEEENSI_IJLi5EEEENSI_IJLi8EEEENSI_IJLi9EEEENSI_IJLi10EEEEEEENS2_IJNSI_IJLi1ELi2ELi3EEEENSI_IJLi4ELi5EEEENSI_IJLi6EEEESO_SQ_SR_SS_NSI_IJLi11ELi12EEEENSI_IJLi13EEEENSI_IJLi14EEEEEEENSI_IJLi11ELi12ELi13ELi14EEEEiEENS_16TensorCoordinateILi15EKS11_EENS_20TensorCoordinateStepILi10ELi4ENSI_IJLi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0EEEEEEEEvRKT_RT0_RKT1_ENKUlS19_E_clINS6_IiLi9EEEEEDaS19_.exit.i.i.i.i.i244.i
                                        ; =>This Inner Loop Header: Depth=1
	v_cmp_le_i32_e32 vcc, s28, v2
	v_cmp_gt_i32_e64 s[0:1], s27, v2
	v_lshlrev_b32_e32 v25, 1, v23
	v_add_u32_e32 v23, s2, v23
	s_and_b64 s[44:45], vcc, s[0:1]
	v_cmp_le_i32_e32 vcc, s25, v3
	v_cmp_gt_i32_e64 s[0:1], s24, v3
	v_lshlrev_b32_e32 v29, 1, v23
	v_add_u32_e32 v23, s2, v23
	s_and_b64 s[0:1], vcc, s[0:1]
	v_lshlrev_b32_e32 v36, 1, v23
	v_add_u32_e32 v23, s2, v23
	s_and_b64 s[0:1], s[0:1], s[44:45]
	v_mov_b32_e32 v44, s26
	v_lshlrev_b32_e32 v40, 1, v23
	v_cndmask_b32_e64 v56, v44, 0, s[0:1]
	buffer_load_dwordx4 v[25:28], v25, s[12:15], 0 offen
	v_lshl_add_u32 v44, v24, 1, v56
	buffer_load_dwordx4 v[29:32], v29, s[12:15], 0 offen
	v_add_u32_e32 v24, s20, v24
	buffer_load_dwordx4 v[36:39], v36, s[12:15], 0 offen
	v_lshl_add_u32 v48, v24, 1, v56
	buffer_load_dwordx4 v[40:43], v40, s[12:15], 0 offen
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[60:63], v7 offset1:1
	v_add_u32_e32 v24, s20, v24
	v_lshl_add_u32 v52, v24, 1, v56
	v_add_u32_e32 v24, s20, v24
	v_lshl_add_u32 v56, v24, 1, v56
	v_add_u32_e32 v64, s34, v7
	buffer_load_dwordx4 v[44:47], v44, s[16:19], 0 offen
	v_add_u32_e32 v72, s34, v5
	buffer_load_dwordx4 v[48:51], v48, s[16:19], 0 offen
	v_add_u32_e32 v80, s34, v8
	buffer_load_dwordx4 v[52:55], v52, s[16:19], 0 offen
	v_add_u32_e32 v88, s34, v10
	buffer_load_dwordx4 v[56:59], v56, s[16:19], 0 offen
	ds_read2_b64 v[64:67], v64 offset0:32 offset1:33
	ds_read2_b64 v[68:71], v5 offset1:1
	ds_read2_b64 v[76:79], v8 offset1:1
	ds_read2_b64 v[84:87], v10 offset1:1
	ds_read2_b64 v[92:95], v11 offset1:1
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8f16 a[192:207], v[60:61], v[68:69], a[192:207]
	v_add_u32_e32 v96, s34, v11
	ds_read2_b64 v[72:75], v72 offset0:32 offset1:33
	ds_read2_b64 v[80:83], v80 offset0:32 offset1:33
	ds_read2_b64 v[88:91], v88 offset0:32 offset1:33
	ds_read2_b64 v[96:99], v96 offset0:32 offset1:33
	v_mul_hi_u32 v101, s10, v16
	v_mul_hi_u32 v100, s4, v17
	v_add_u32_e32 v1, 4, v1
	v_add_u32_e32 v17, 4, v17
	v_add_u32_e32 v16, 4, v16
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8f16 a[224:239], v[60:61], v[76:77], a[224:239]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8f16 a[208:223], v[60:61], v[84:85], a[208:223]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8f16 a[176:191], v[60:61], v[92:93], a[176:191]
	v_mfma_f32_32x32x8f16 a[192:207], v[62:63], v[70:71], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[62:63], v[78:79], a[224:239]
	v_mfma_f32_32x32x8f16 a[208:223], v[62:63], v[86:87], a[208:223]
	v_mfma_f32_32x32x8f16 a[176:191], v[62:63], v[94:95], a[176:191]
	ds_read2_b64 v[60:63], v13 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[160:175], v[60:61], v[68:69], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[60:61], v[76:77], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[60:61], v[84:85], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[60:61], v[92:93], a[112:127]
	v_mfma_f32_32x32x8f16 a[192:207], v[64:65], v[72:73], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[64:65], v[80:81], a[224:239]
	v_mfma_f32_32x32x8f16 a[208:223], v[64:65], v[88:89], a[208:223]
	v_mfma_f32_32x32x8f16 a[176:191], v[64:65], v[96:97], a[176:191]
	v_add_u32_e32 v64, s34, v13
	v_mfma_f32_32x32x8f16 a[160:175], v[62:63], v[70:71], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[62:63], v[78:79], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[62:63], v[86:87], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[62:63], v[94:95], a[112:127]
	ds_read2_b64 v[60:63], v12 offset1:1
	v_mfma_f32_32x32x8f16 a[192:207], v[66:67], v[74:75], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[66:67], v[82:83], a[224:239]
	v_mfma_f32_32x32x8f16 a[208:223], v[66:67], v[90:91], a[208:223]
	v_mfma_f32_32x32x8f16 a[176:191], v[66:67], v[98:99], a[176:191]
	ds_read2_b64 v[64:67], v64 offset0:32 offset1:33
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[96:111], v[60:61], v[68:69], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[60:61], v[76:77], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[60:61], v[84:85], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[60:61], v[92:93], a[48:63]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[160:175], v[64:65], v[72:73], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[64:65], v[80:81], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[64:65], v[88:89], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[64:65], v[96:97], a[112:127]
	v_add_u32_e32 v64, s34, v12
	v_mfma_f32_32x32x8f16 a[96:111], v[62:63], v[70:71], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[62:63], v[78:79], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[62:63], v[86:87], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[62:63], v[94:95], a[48:63]
	ds_read2_b64 v[60:63], v9 offset1:1
	v_mfma_f32_32x32x8f16 a[160:175], v[66:67], v[74:75], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[66:67], v[82:83], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[66:67], v[90:91], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[66:67], v[98:99], a[112:127]
	ds_read2_b64 v[64:67], v64 offset0:32 offset1:33
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[32:47], v[60:61], v[68:69], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[60:61], v[76:77], a[16:31]
	v_mfma_f32_32x32x8f16 a[0:15], v[60:61], v[84:85], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[60:61], v[92:93], a[240:255]
	v_add3_u32 v60, v15, v101, s43
	v_lshrrev_b32_e32 v60, s11, v60
	v_mul_lo_u32 v61, s33, v60
	v_sub_u32_e32 v22, v60, v22
	v_lshl_add_u32 v22, v22, 3, -3
	v_mul_lo_u32 v22, v22, s2
	v_sub_u32_e32 v21, v61, v21
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[96:111], v[64:65], v[72:73], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[64:65], v[80:81], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[64:65], v[88:89], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[64:65], v[96:97], a[48:63]
	v_add_u32_e32 v64, s34, v9
	v_mfma_f32_32x32x8f16 a[32:47], v[62:63], v[70:71], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[62:63], v[78:79], a[16:31]
	v_mfma_f32_32x32x8f16 a[0:15], v[62:63], v[86:87], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[62:63], v[94:95], a[240:255]
	v_add_u32_e32 v62, s43, v15
	v_add_u32_e32 v21, v62, v21
	v_mul_lo_u32 v21, v21, s3
	v_add_u32_e32 v63, v62, v61
	v_add3_u32 v23, v22, v23, v21
	v_add3_u32 v21, v14, v100, s43
	v_lshrrev_b32_e32 v21, s41, v21
	v_mul_lo_u32 v61, s30, v21
	v_mul_lo_u32 v22, v21, s37
	v_sub_u32_e32 v20, v61, v20
	v_mul_hi_u32 v61, v21, s38
	v_mfma_f32_32x32x8f16 a[96:111], v[66:67], v[74:75], a[96:111]
	v_add3_u32 v20, v14, s43, v20
	v_mul_lo_u32 v20, v20, s35
	v_add_u32_e32 v61, v21, v61
	v_lshrrev_b32_e32 v61, s40, v61
	v_mul_lo_u32 v62, v61, s36
	v_sub_u32_e32 v18, v61, v18
	v_lshl_add_u32 v18, v18, 3, -3
	v_add_u32_e32 v2, v20, v2
	v_sub_u32_e32 v62, v21, v62
	v_sub_u32_e32 v19, v62, v19
	v_mul_lo_u32 v19, v19, s31
	v_mul_lo_u32 v20, v20, s22
	v_mul_lo_u32 v18, v18, s20
	v_sub_u32_e32 v22, v1, v22
	v_add_u32_e32 v3, v19, v3
	v_mfma_f32_32x32x8f16 a[80:95], v[66:67], v[82:83], a[80:95]
	v_mul_lo_u32 v19, v19, s21
	v_add_u32_e32 v20, v20, v24
	s_add_i32 s43, s43, 4
	s_cmp_lt_i32 s43, s29
	v_add3_u32 v24, v20, v18, v19
	v_mfma_f32_32x32x8f16 a[64:79], v[66:67], v[90:91], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[66:67], v[98:99], a[48:63]
	ds_read2_b64 v[64:67], v64 offset0:32 offset1:33
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	
             v_pack_b32_f16 v18, v25, v29 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v20, v25, v29, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v19, v36, v40 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v21, v36, v40, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v25, v26, v30 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v29, v26, v30, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v26, v37, v41 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v30, v37, v41, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v36, v27, v31 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v40, v27, v31, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v37, v38, v42 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v41, v38, v42, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v27, v28, v32 
             
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[32:47], v[64:65], v[72:73], a[32:47]
	;;#ASMSTART
	
             v_pack_b32_f16 v31, v28, v32, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v28, v39, v43 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v32, v39, v43, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v4, v[18:19], v[20:21] offset1:2
	ds_write2_b64 v4, v[25:26], v[29:30] offset0:4 offset1:6
	ds_write2_b64 v4, v[36:37], v[40:41] offset0:8 offset1:10
	ds_write2_b64 v4, v[27:28], v[31:32] offset0:12 offset1:14
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v18, v44, v48 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v20, v44, v48, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v19, v52, v56 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v21, v52, v56, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v25, v45, v49 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v27, v45, v49, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v26, v53, v57 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v28, v53, v57, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v29, v46, v50 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v31, v46, v50, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v30, v54, v58 
             
	;;#ASMEND
	v_mfma_f32_32x32x8f16 a[16:31], v[64:65], v[80:81], a[16:31]
	;;#ASMSTART
	
             v_pack_b32_f16 v32, v54, v58, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v36, v47, v51 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v38, v47, v51, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v37, v55, v59 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v39, v55, v59, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v6, v[18:19], v[20:21] offset1:2
	ds_write2_b64 v6, v[25:26], v[27:28] offset0:4 offset1:6
	ds_write2_b64 v6, v[29:30], v[31:32] offset0:8 offset1:10
	ds_write2_b64 v6, v[36:37], v[38:39] offset0:12 offset1:14
	v_mov_b32_e32 v18, v61
	v_mov_b32_e32 v19, v62
	v_mov_b32_e32 v20, v22
	v_mov_b32_e32 v21, v63
	v_mov_b32_e32 v22, v60
	v_mfma_f32_32x32x8f16 a[0:15], v[64:65], v[88:89], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[64:65], v[96:97], a[240:255]
	v_mfma_f32_32x32x8f16 a[32:47], v[66:67], v[74:75], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[66:67], v[82:83], a[16:31]
	v_mfma_f32_32x32x8f16 a[0:15], v[66:67], v[90:91], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[66:67], v[98:99], a[240:255]
	s_cbranch_scc1 BB1_1
; %bb.2:                                ; %_ZZN2ck23Merge_v2_magic_divisionINS_5TupleIJNS_17integral_constantIiLi4EEENS2_IiLi2EEEiiiEEEEC1ERKS5_ENKUlT_E_clIS4_EEDaS9_.exit.i.i.i.i.i.i.i.i
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[1:4], v7 offset1:1
	s_movk_i32 s0, 0x1000
	v_add_u32_e32 v6, s0, v7
	ds_read2_b64 v[36:39], v6 offset0:32 offset1:33
	ds_read2_b64 v[14:17], v5 offset1:1
	ds_read2_b64 v[18:21], v8 offset1:1
	ds_read2_b64 v[22:25], v10 offset1:1
	ds_read2_b64 v[26:29], v11 offset1:1
	v_add_u32_e32 v5, s0, v5
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8f16 a[192:207], v[1:2], v[14:15], a[192:207]
	ds_read2_b64 v[40:43], v5 offset0:32 offset1:33
	v_add_u32_e32 v5, s0, v8
	ds_read2_b64 v[5:8], v5 offset0:32 offset1:33
	v_add_u32_e32 v10, s0, v10
	ds_read2_b64 v[44:47], v10 offset0:32 offset1:33
	v_add_u32_e32 v10, s0, v11
	ds_read2_b64 v[48:51], v10 offset0:32 offset1:33
	v_add_u32_e32 v10, s0, v13
	ds_read2_b64 v[52:55], v10 offset0:32 offset1:33
	v_add_u32_e32 v10, s0, v12
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8f16 a[224:239], v[1:2], v[18:19], a[224:239]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8f16 a[208:223], v[1:2], v[22:23], a[208:223]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8f16 a[176:191], v[1:2], v[26:27], a[176:191]
	v_mfma_f32_32x32x8f16 a[192:207], v[3:4], v[16:17], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[3:4], v[20:21], a[224:239]
	v_mfma_f32_32x32x8f16 a[208:223], v[3:4], v[24:25], a[208:223]
	v_mfma_f32_32x32x8f16 a[176:191], v[3:4], v[28:29], a[176:191]
	ds_read2_b64 v[1:4], v13 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[160:175], v[1:2], v[14:15], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[1:2], v[18:19], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[1:2], v[22:23], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[1:2], v[26:27], a[112:127]
	v_mfma_f32_32x32x8f16 a[160:175], v[3:4], v[16:17], a[160:175]
	v_mfma_f32_32x32x8f16 a[144:159], v[3:4], v[20:21], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[3:4], v[24:25], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[3:4], v[28:29], a[112:127]
	ds_read2_b64 v[1:4], v12 offset1:1
	ds_read2_b64 v[10:13], v10 offset0:32 offset1:33
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[96:111], v[1:2], v[14:15], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[1:2], v[18:19], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[1:2], v[22:23], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[26:27], a[48:63]
	v_mfma_f32_32x32x8f16 a[96:111], v[3:4], v[16:17], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[3:4], v[20:21], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[3:4], v[24:25], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[28:29], a[48:63]
	ds_read2_b64 v[1:4], v9 offset1:1
	v_add_u32_e32 v9, s0, v9
	ds_read2_b64 v[56:59], v9 offset0:32 offset1:33
	s_movk_i32 s0, 0x80
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[18:19], a[16:31]
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[20:21], a[16:31]
	v_mfma_f32_32x32x8f16 a[32:47], v[1:2], v[14:15], a[32:47]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[16:31], v[56:57], v[5:6], a[16:31]
	v_mfma_f32_32x32x8f16 a[0:15], v[1:2], v[22:23], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[1:2], v[26:27], a[240:255]
	v_mfma_f32_32x32x8f16 a[192:207], v[36:37], v[40:41], a[192:207]
	v_mfma_f32_32x32x8f16 a[32:47], v[3:4], v[16:17], a[32:47]
	v_mfma_f32_32x32x8f16 a[0:15], v[3:4], v[24:25], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[3:4], v[28:29], a[240:255]
	v_mfma_f32_32x32x8f16 a[16:31], v[58:59], v[7:8], a[16:31]
	v_mfma_f32_32x32x8f16 a[192:207], v[38:39], v[42:43], a[192:207]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v3, a16              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:4 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a17              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:8 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a18              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:12 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a19              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[224:239], v[36:37], v[5:6], a[224:239]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:16 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a20              ;  Reload Reuse
	v_accvgpr_read_b32 v17, a192
	v_accvgpr_read_b32 v18, a193
	buffer_store_dword v3, off, s[64:67], 0 offset:20 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a21              ;  Reload Reuse
	v_accvgpr_read_b32 v19, a194
	v_accvgpr_read_b32 v20, a195
	buffer_store_dword v3, off, s[64:67], 0 offset:24 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a22              ;  Reload Reuse
	v_accvgpr_read_b32 v21, a196
	v_accvgpr_read_b32 v22, a197
	buffer_store_dword v3, off, s[64:67], 0 offset:28 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a23              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[208:223], v[36:37], v[44:45], a[208:223]
	v_accvgpr_read_b32 v23, a198
	buffer_store_dword v3, off, s[64:67], 0 offset:32 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a24              ;  Reload Reuse
	v_accvgpr_read_b32 v24, a199
	v_accvgpr_read_b32 v25, a200
	buffer_store_dword v3, off, s[64:67], 0 offset:36 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a25              ;  Reload Reuse
	v_accvgpr_read_b32 v26, a201
	v_accvgpr_read_b32 v27, a202
	buffer_store_dword v3, off, s[64:67], 0 offset:40 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a26              ;  Reload Reuse
	v_accvgpr_read_b32 v28, a203
	v_accvgpr_read_b32 v29, a204
	buffer_store_dword v3, off, s[64:67], 0 offset:44 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a27              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[176:191], v[36:37], v[48:49], a[176:191]
	v_accvgpr_read_b32 v30, a205
	buffer_store_dword v3, off, s[64:67], 0 offset:48 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a28              ;  Reload Reuse
	v_accvgpr_read_b32 v31, a206
	v_accvgpr_read_b32 v32, a207
	buffer_store_dword v3, off, s[64:67], 0 offset:52 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a29              ;  Reload Reuse
	v_mul_i32_i24_e32 v36, 0xffffffe0, v33
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:56 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a30              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:60 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a31              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[160:175], v[52:53], v[40:41], a[160:175]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:64 ; 4-byte Folded Spill
	v_mfma_f32_32x32x8f16 a[144:159], v[52:53], v[5:6], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[52:53], v[44:45], a[128:143]
	v_mfma_f32_32x32x8f16 a[112:127], v[52:53], v[48:49], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[10:11], v[40:41], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[10:11], v[5:6], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[10:11], v[44:45], a[64:79]
	v_mfma_f32_32x32x8f16 a[48:63], v[10:11], v[48:49], a[48:63]
	v_mfma_f32_32x32x8f16 a[32:47], v[56:57], v[40:41], a[32:47]
	v_mfma_f32_32x32x8f16 a[0:15], v[56:57], v[44:45], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[56:57], v[48:49], a[240:255]
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mfma_f32_32x32x8f16 a[224:239], v[38:39], v[7:8], a[224:239]
	v_mfma_f32_32x32x8f16 a[176:191], v[38:39], v[50:51], a[176:191]
	v_mfma_f32_32x32x8f16 a[144:159], v[54:55], v[7:8], a[144:159]
	v_mfma_f32_32x32x8f16 a[112:127], v[54:55], v[50:51], a[112:127]
	v_mfma_f32_32x32x8f16 a[80:95], v[12:13], v[7:8], a[80:95]
	v_mfma_f32_32x32x8f16 a[48:63], v[12:13], v[50:51], a[48:63]
	v_mfma_f32_32x32x8f16 a[16:31], v[58:59], v[50:51], a[240:255]
	v_mfma_f32_32x32x8f16 a[192:207], v[54:55], v[42:43], a[160:175]
	v_mfma_f32_32x32x8f16 a[240:255], v[12:13], v[42:43], a[96:111]
	v_mfma_f32_32x32x8f16 a[96:111], v[58:59], v[42:43], a[32:47]
	v_mfma_f32_32x32x8f16 a[208:223], v[38:39], v[46:47], a[208:223]
	v_mfma_f32_32x32x8f16 a[64:79], v[12:13], v[46:47], a[64:79]
	v_mfma_f32_32x32x8f16 a[160:175], v[54:55], v[46:47], a[128:143]
	v_mfma_f32_32x32x8f16 a[32:47], v[58:59], v[46:47], a[0:15]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_4
; %bb.3:
	v_lshrrev_b32_e32 v1, 2, v0
	v_mul_i32_i24_e32 v2, -4, v1
	v_add_u32_e32 v1, v36, v1
	v_lshlrev_b32_e32 v3, 1, v1
	v_add_u32_e32 v4, s6, v33
	v_lshl_add_u32 v3, v4, 8, v3
	v_mul_lo_u32 v3, v3, s7
	v_add_lshl_u32 v2, v2, v0, 4
	v_lshlrev_b32_e32 v4, 12, v33
	v_lshlrev_b32_e32 v1, 7, v1
	v_add3_u32 v49, v2, v4, v1
	v_add3_u32 v48, s5, v2, v3
BB1_4:                                  ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EEC2ERSO_RKNSA_IJiiiiEEES15_S1A_RKS3_.exit.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_or_b32 v0, v0, 4, v34
	v_lshlrev_b32_e32 v33, 5, v33
	v_lshrrev_b32_e32 v34, 6, v35
	v_add3_u32 v0, v0, v36, v33
	v_sub_u32_e32 v0, v0, v34
	v_lshlrev_b32_e32 v0, 6, v0
	v_cvt_f16_f32_e32 v17, v17
	v_add_lshl_u32 v50, v0, v35, 1
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v18, v19
	v_cvt_f16_f32_e32 v19, v20
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v17
	ds_write_b16 v50, v0 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v17, v23
	v_cvt_f16_f32_e32 v18, v22
	v_cvt_f16_f32_e32 v19, v21
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v17, v26
	v_cvt_f16_f32_e32 v18, v27
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v17, v31
	v_accvgpr_read_b32 v1, a224
	v_cvt_f16_f32_e32 v18, v30
	v_accvgpr_read_b32 v2, a225
	v_accvgpr_read_b32 v3, a226
	v_accvgpr_read_b32 v4, a227
	v_accvgpr_read_b32 v5, a228
	v_accvgpr_read_b32 v6, a229
	v_accvgpr_read_b32 v7, a230
	v_accvgpr_read_b32 v8, a231
	v_accvgpr_read_b32 v9, a232
	v_accvgpr_read_b32 v10, a233
	v_accvgpr_read_b32 v11, a234
	v_accvgpr_read_b32 v12, a235
	v_accvgpr_read_b32 v13, a236
	v_accvgpr_read_b32 v14, a237
	v_accvgpr_read_b32 v15, a238
	v_accvgpr_read_b32 v16, a239
	v_cvt_f16_f32_e32 v19, v29
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_6
; %bb.5:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i.i.i.i
	v_lshlrev_b32_e32 v0, 1, v49
	ds_read2_b64 v[17:20], v0 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v20, v21, s[8:11], 12 offen
	ds_read2_b64 v[17:20], v0 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v20, v22, s[8:11], 12 offen
	ds_read2_b64 v[17:20], v0 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v20, v21, s[8:11], 12 offen
	ds_read2_b64 v[17:20], v0 offset0:16 offset1:17
	v_add_lshl_u32 v0, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v17, v0, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v18, v0, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v19, v0, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v20, v0, s[8:11], 12 offen
BB1_6:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v1, v2
	v_cvt_f16_f32_e32 v2, v3
	v_cvt_f16_f32_e32 v3, v4
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v7
	v_cvt_f16_f32_e32 v2, v6
	v_cvt_f16_f32_e32 v3, v5
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v9
	v_cvt_f16_f32_e32 v1, v10
	v_cvt_f16_f32_e32 v2, v11
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v15
	v_accvgpr_read_b32 v32, a208
	v_cvt_f16_f32_e32 v2, v14
	v_accvgpr_read_b32 v33, a209
	v_accvgpr_read_b32 v34, a210
	v_accvgpr_read_b32 v35, a211
	v_accvgpr_read_b32 v36, a212
	v_accvgpr_read_b32 v37, a213
	v_accvgpr_read_b32 v38, a214
	v_accvgpr_read_b32 v39, a215
	v_accvgpr_read_b32 v40, a216
	v_accvgpr_read_b32 v41, a217
	v_accvgpr_read_b32 v42, a218
	v_accvgpr_read_b32 v43, a219
	v_accvgpr_read_b32 v44, a220
	v_accvgpr_read_b32 v45, a221
	v_accvgpr_read_b32 v46, a222
	v_accvgpr_read_b32 v47, a223
	v_cvt_f16_f32_e32 v3, v13
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_8
; %bb.7:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i97.i.i.i.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 12 offen
BB1_8:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_122.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v1, v33
	v_cvt_f16_f32_e32 v2, v34
	v_cvt_f16_f32_e32 v3, v35
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v39
	v_cvt_f16_f32_e32 v1, v38
	v_cvt_f16_f32_e32 v2, v37
	v_cvt_f16_f32_e32 v3, v36
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v40
	v_cvt_f16_f32_e32 v1, v41
	v_cvt_f16_f32_e32 v2, v42
	v_cvt_f16_f32_e32 v3, v43
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v47
	v_cvt_f16_f32_e32 v1, v46
	v_accvgpr_read_b32 v16, a176
	v_cvt_f16_f32_e32 v2, v45
	v_accvgpr_read_b32 v17, a177
	v_accvgpr_read_b32 v18, a178
	v_accvgpr_read_b32 v19, a179
	v_accvgpr_read_b32 v20, a180
	v_accvgpr_read_b32 v21, a181
	v_accvgpr_read_b32 v22, a182
	v_accvgpr_read_b32 v23, a183
	v_accvgpr_read_b32 v24, a184
	v_accvgpr_read_b32 v25, a185
	v_accvgpr_read_b32 v26, a186
	v_accvgpr_read_b32 v27, a187
	v_accvgpr_read_b32 v28, a188
	v_accvgpr_read_b32 v29, a189
	v_accvgpr_read_b32 v30, a190
	v_accvgpr_read_b32 v31, a191
	v_cvt_f16_f32_e32 v3, v44
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_10
; %bb.9:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i219.i.i.i.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 12 offen
BB1_10:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_244.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a112
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a113
	v_accvgpr_read_b32 v2, a114
	v_accvgpr_read_b32 v3, a115
	v_accvgpr_read_b32 v4, a116
	v_accvgpr_read_b32 v5, a117
	v_accvgpr_read_b32 v6, a118
	v_accvgpr_read_b32 v7, a119
	v_accvgpr_read_b32 v8, a120
	v_accvgpr_read_b32 v9, a121
	v_accvgpr_read_b32 v10, a122
	v_accvgpr_read_b32 v11, a123
	v_accvgpr_read_b32 v12, a124
	v_accvgpr_read_b32 v13, a125
	v_accvgpr_read_b32 v14, a126
	v_accvgpr_read_b32 v15, a127
	v_cvt_f16_f32_e32 v19, v28
	s_mul_i32 s2, s7, 63
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_12
; %bb.11:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_u32_e32 v21, s7, v48
	v_lshlrev_b32_e32 v20, 1, v21
	v_add_u32_e32 v48, s2, v21
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[8:11], 12 offen
BB1_12:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a160
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a161
	v_accvgpr_read_b32 v18, a162
	v_accvgpr_read_b32 v19, a163
	v_accvgpr_read_b32 v20, a164
	v_accvgpr_read_b32 v21, a165
	v_accvgpr_read_b32 v22, a166
	v_accvgpr_read_b32 v23, a167
	v_accvgpr_read_b32 v24, a168
	v_accvgpr_read_b32 v25, a169
	v_accvgpr_read_b32 v26, a170
	v_accvgpr_read_b32 v27, a171
	v_accvgpr_read_b32 v28, a172
	v_accvgpr_read_b32 v29, a173
	v_accvgpr_read_b32 v30, a174
	v_accvgpr_read_b32 v31, a175
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_14
; %bb.13:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i108.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 12 offen
BB1_14:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i156.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a144
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a145
	v_accvgpr_read_b32 v2, a146
	v_accvgpr_read_b32 v3, a147
	v_accvgpr_read_b32 v4, a148
	v_accvgpr_read_b32 v5, a149
	v_accvgpr_read_b32 v6, a150
	v_accvgpr_read_b32 v7, a151
	v_accvgpr_read_b32 v8, a152
	v_accvgpr_read_b32 v9, a153
	v_accvgpr_read_b32 v10, a154
	v_accvgpr_read_b32 v11, a155
	v_accvgpr_read_b32 v12, a156
	v_accvgpr_read_b32 v13, a157
	v_accvgpr_read_b32 v14, a158
	v_accvgpr_read_b32 v15, a159
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_16
; %bb.15:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i97.i.i.i224.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_lshl_u32 v20, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[8:11], 12 offen
BB1_16:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_122.i.i.i272.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a192
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a193
	v_accvgpr_read_b32 v18, a194
	v_accvgpr_read_b32 v19, a195
	v_accvgpr_read_b32 v20, a196
	v_accvgpr_read_b32 v21, a197
	v_accvgpr_read_b32 v22, a198
	v_accvgpr_read_b32 v23, a199
	v_accvgpr_read_b32 v24, a200
	v_accvgpr_read_b32 v25, a201
	v_accvgpr_read_b32 v26, a202
	v_accvgpr_read_b32 v27, a203
	v_accvgpr_read_b32 v28, a204
	v_accvgpr_read_b32 v29, a205
	v_accvgpr_read_b32 v30, a206
	v_accvgpr_read_b32 v31, a207
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_18
; %bb.17:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i219.i.i.i340.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 12 offen
BB1_18:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_244.i.i.i388.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a240
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a241
	v_accvgpr_read_b32 v2, a242
	v_accvgpr_read_b32 v3, a243
	v_accvgpr_read_b32 v4, a244
	v_accvgpr_read_b32 v5, a245
	v_accvgpr_read_b32 v6, a246
	v_accvgpr_read_b32 v7, a247
	v_accvgpr_read_b32 v8, a248
	v_accvgpr_read_b32 v9, a249
	v_accvgpr_read_b32 v10, a250
	v_accvgpr_read_b32 v11, a251
	v_accvgpr_read_b32 v12, a252
	v_accvgpr_read_b32 v13, a253
	v_accvgpr_read_b32 v14, a254
	v_accvgpr_read_b32 v15, a255
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_20
; %bb.19:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i444.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_u32_e32 v21, s7, v48
	v_lshlrev_b32_e32 v20, 1, v21
	v_add_u32_e32 v48, s2, v21
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[8:11], 12 offen
BB1_20:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I469.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a80
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a81
	v_accvgpr_read_b32 v18, a82
	v_accvgpr_read_b32 v19, a83
	v_accvgpr_read_b32 v20, a84
	v_accvgpr_read_b32 v21, a85
	v_accvgpr_read_b32 v22, a86
	v_accvgpr_read_b32 v23, a87
	v_accvgpr_read_b32 v24, a88
	v_accvgpr_read_b32 v25, a89
	v_accvgpr_read_b32 v26, a90
	v_accvgpr_read_b32 v27, a91
	v_accvgpr_read_b32 v28, a92
	v_accvgpr_read_b32 v29, a93
	v_accvgpr_read_b32 v30, a94
	v_accvgpr_read_b32 v31, a95
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_22
; %bb.21:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i577.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 12 offen
BB1_22:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i625.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a64
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a65
	v_accvgpr_read_b32 v2, a66
	v_accvgpr_read_b32 v3, a67
	v_accvgpr_read_b32 v4, a68
	v_accvgpr_read_b32 v5, a69
	v_accvgpr_read_b32 v6, a70
	v_accvgpr_read_b32 v7, a71
	v_accvgpr_read_b32 v8, a72
	v_accvgpr_read_b32 v9, a73
	v_accvgpr_read_b32 v10, a74
	v_accvgpr_read_b32 v11, a75
	v_accvgpr_read_b32 v12, a76
	v_accvgpr_read_b32 v13, a77
	v_accvgpr_read_b32 v14, a78
	v_accvgpr_read_b32 v15, a79
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_24
; %bb.23:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i97.i.i.i693.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_lshl_u32 v20, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[8:11], 12 offen
BB1_24:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_122.i.i.i741.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a48
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a49
	v_accvgpr_read_b32 v18, a50
	v_accvgpr_read_b32 v19, a51
	v_accvgpr_read_b32 v20, a52
	v_accvgpr_read_b32 v21, a53
	v_accvgpr_read_b32 v22, a54
	v_accvgpr_read_b32 v23, a55
	v_accvgpr_read_b32 v24, a56
	v_accvgpr_read_b32 v25, a57
	v_accvgpr_read_b32 v26, a58
	v_accvgpr_read_b32 v27, a59
	v_accvgpr_read_b32 v28, a60
	v_accvgpr_read_b32 v29, a61
	v_accvgpr_read_b32 v30, a62
	v_accvgpr_read_b32 v31, a63
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_26
; %bb.25:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i219.i.i.i809.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s7, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 12 offen
BB1_26:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_244.i.i.i857.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a16
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a17
	v_accvgpr_read_b32 v2, a18
	v_accvgpr_read_b32 v3, a19
	v_accvgpr_read_b32 v4, a20
	v_accvgpr_read_b32 v5, a21
	v_accvgpr_read_b32 v6, a22
	v_accvgpr_read_b32 v7, a23
	v_accvgpr_read_b32 v8, a24
	v_accvgpr_read_b32 v9, a25
	v_accvgpr_read_b32 v10, a26
	v_accvgpr_read_b32 v11, a27
	v_accvgpr_read_b32 v12, a28
	v_accvgpr_read_b32 v13, a29
	v_accvgpr_read_b32 v14, a30
	v_accvgpr_read_b32 v15, a31
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_28
; %bb.27:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i913.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_u32_e32 v21, s7, v48
	v_lshlrev_b32_e32 v20, 1, v21
	v_add_u32_e32 v48, s2, v21
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[8:11], 12 offen
BB1_28:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I938.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a32
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a33
	v_accvgpr_read_b32 v18, a34
	v_accvgpr_read_b32 v19, a35
	v_accvgpr_read_b32 v20, a36
	v_accvgpr_read_b32 v21, a37
	v_accvgpr_read_b32 v22, a38
	v_accvgpr_read_b32 v23, a39
	v_accvgpr_read_b32 v24, a40
	v_accvgpr_read_b32 v25, a41
	v_accvgpr_read_b32 v26, a42
	v_accvgpr_read_b32 v27, a43
	v_accvgpr_read_b32 v28, a44
	v_accvgpr_read_b32 v29, a45
	v_accvgpr_read_b32 v30, a46
	v_accvgpr_read_b32 v31, a47
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_30
; %bb.29:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i1046.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 12 offen
BB1_30:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i1094.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v32, off, s[64:67], 0 offset:4 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:64 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_cvt_f16_f32_e32 v18, v29
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v32            ;  Reload Reuse
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v15, a15
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_32
; %bb.31:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i97.i.i.i1162.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_lshl_u32 v20, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[8:11], 12 offen
BB1_32:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_122.i.i.i1210.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a96
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a97
	v_accvgpr_read_b32 v18, a98
	v_accvgpr_read_b32 v19, a99
	v_accvgpr_read_b32 v20, a100
	v_accvgpr_read_b32 v21, a101
	v_accvgpr_read_b32 v22, a102
	v_accvgpr_read_b32 v23, a103
	v_accvgpr_read_b32 v24, a104
	v_accvgpr_read_b32 v25, a105
	v_accvgpr_read_b32 v26, a106
	v_accvgpr_read_b32 v27, a107
	v_accvgpr_read_b32 v28, a108
	v_accvgpr_read_b32 v29, a109
	v_accvgpr_read_b32 v30, a110
	v_accvgpr_read_b32 v31, a111
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_34
; %bb.33:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i219.i.i.i1278.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s7, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 12 offen
BB1_34:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_244.i.i.i1326.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v17
	v_cvt_f16_f32_e32 v2, v18
	v_cvt_f16_f32_e32 v3, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v23
	v_cvt_f16_f32_e32 v1, v22
	v_cvt_f16_f32_e32 v2, v21
	v_cvt_f16_f32_e32 v3, v20
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v25
	v_cvt_f16_f32_e32 v2, v26
	v_cvt_f16_f32_e32 v3, v27
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v31
	v_cvt_f16_f32_e32 v1, v30
	v_cvt_f16_f32_e32 v2, v29
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_36
; %bb.35:
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 12 offen
BB1_36:                                 ; %_ZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_IJ
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
		.amdhsa_group_segment_fixed_size 34816
		.amdhsa_private_segment_fixed_size 68
		.amdhsa_kernarg_size 600
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 68
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_, .Lfunc_end1-_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 16496
; NumSgprs: 70
; NumVgprs: 102
; NumAgprs: 256
; TotalNumVgprs: 256
; ScratchSize: 68
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 34816 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 70
; NumVGPRsForWavesPerEU: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_ ; -- Begin function _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
	.globl	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
	.p2align	8
	.type	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_,@function
_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_: ; @_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
; %bb.0:
	s_mov_b64 s[58:59], s[2:3]
	s_mov_b64 s[56:57], s[0:1]
	s_add_u32 s56, s56, s7
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_load_dwordx2 s[12:13], s[4:5], 0x8
	s_load_dword s24, s[4:5], 0x48
	s_load_dword s52, s[4:5], 0x50
	s_load_dword s25, s[4:5], 0x58
	s_load_dword s53, s[4:5], 0x70
	s_load_dword s7, s[4:5], 0x84
	s_load_dwordx4 s[16:19], s[4:5], 0x98
	s_load_dwordx4 s[20:23], s[4:5], 0xac
	s_load_dwordx2 s[10:11], s[4:5], 0xbc
	s_load_dwordx2 s[2:3], s[4:5], 0xd4
	s_load_dwordx2 s[8:9], s[4:5], 0xe4
	s_load_dwordx2 s[26:27], s[4:5], 0x114
	s_load_dwordx2 s[30:31], s[4:5], 0x120
	s_load_dwordx2 s[28:29], s[4:5], 0x12c
	s_load_dwordx2 s[14:15], s[4:5], 0x13c
	s_load_dwordx2 s[36:37], s[4:5], 0x148
	s_load_dwordx2 s[34:35], s[4:5], 0x154
	s_load_dword s33, s[4:5], 0x16c
	s_waitcnt lgkmcnt(0)
	s_load_dword s19, s[4:5], 0x180
	s_load_dword s54, s[4:5], 0x1d4
	s_load_dwordx4 s[40:43], s[4:5], 0x1e0
	s_load_dwordx4 s[44:47], s[4:5], 0x1f4
	s_load_dwordx4 s[48:51], s[4:5], 0x208
	s_addc_u32 s57, s57, 0
	v_lshrrev_b32_e32 v1, 5, v0
	v_lshrrev_b32_e32 v33, 7, v0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s38, s47, s6
	s_add_i32 s38, s6, s38
	s_lshr_b32 s38, s38, s51
	s_mul_i32 s39, s38, s43
	s_sub_i32 s6, s6, s39
	s_mul_hi_u32 s39, s38, s46
	s_add_i32 s39, s38, s39
	s_lshr_b32 s43, s39, s50
	s_mul_i32 s39, s43, s42
	s_sub_i32 s39, s38, s39
	s_mul_hi_u32 s38, s43, s45
	s_add_i32 s38, s43, s38
	s_lshr_b32 s42, s38, s49
	s_mul_i32 s38, s42, s41
	s_sub_i32 s43, s43, s38
	s_mul_hi_u32 s38, s42, s44
	s_add_i32 s38, s42, s38
	s_lshr_b32 s38, s38, s48
	s_mul_i32 s40, s38, s40
	s_mul_i32 s43, s43, s54
	s_sub_i32 s41, s42, s40
	s_add_i32 s40, s6, s43
	v_mad_i32_i24 v2, v33, -4, v1
	s_mul_i32 s6, s38, s53
	v_add_u32_e32 v3, s6, v2
	v_mul_hi_u32 v4, v3, s52
	s_load_dword s42, s[4:5], 0x1c4
	s_load_dword s6, s[4:5], 0x18c
	v_lshlrev_b32_e32 v6, 2, v33
	s_mul_i32 s38, s38, s33
	v_add_u32_e32 v4, v3, v4
	v_lshrrev_b32_e32 v4, s25, v4
	v_mul_lo_u32 v5, v4, s24
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s41, s41, s42
	s_load_dwordx2 s[42:43], s[4:5], 0x24
	s_load_dwordx2 s[24:25], s[4:5], 0x10
	v_lshl_or_b32 v4, v4, 3, v6
	v_sub_u32_e32 v3, v3, v5
	s_add_i32 s41, s41, s39
	s_lshl_b32 s39, s40, 8
	s_movk_i32 s40, 0xffe0
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v4, v4, s42
	v_mul_lo_u32 v3, v3, s43
	v_mad_i32_i24 v1, v1, s40, v0
	s_lshl_b32 s44, s41, 8
	v_lshlrev_b32_e32 v5, 3, v1
	v_add_u32_e32 v7, s44, v5
	v_add3_u32 v3, v7, v4, v3
	v_add_u32_e32 v4, s39, v5
	v_mul_hi_u32 v5, v4, s37
	v_add_u32_e32 v7, s38, v2
	v_mul_hi_u32 v9, v7, s31
	s_movk_i32 s31, 0x44
	v_add_u32_e32 v5, v4, v5
	v_lshrrev_b32_e32 v5, s35, v5
	v_add_u32_e32 v9, v7, v9
	v_mul_hi_u32 v8, v5, s36
	v_lshrrev_b32_e32 v9, s29, v9
	v_mul_hi_u32 v10, v9, s30
	v_mul_lo_u32 v12, v9, s27
	v_add_u32_e32 v8, v5, v8
	v_lshrrev_b32_e32 v8, s34, v8
	v_add_u32_e32 v10, v9, v10
	v_mul_lo_u32 v11, v8, s14
	v_lshrrev_b32_e32 v10, s28, v10
	v_mul_lo_u32 v13, v10, s26
	v_mul_lo_u32 v40, v1, s31
	v_sub_u32_e32 v1, v5, v11
	v_sub_u32_e32 v7, v7, v12
	v_sub_u32_e32 v9, v9, v13
	v_mul_lo_u32 v1, v1, s8
	v_mul_lo_u32 v7, v7, s9
	v_mul_lo_u32 v8, v8, s2
	v_mul_lo_u32 v9, v9, s3
	s_movk_i32 s37, 0x880
	v_add_u32_e32 v17, v7, v1
	v_mul_lo_u32 v2, v2, s37
	v_mul_lo_u32 v5, v5, s15
	v_add_u32_e32 v18, v9, v8
	v_subrev_u32_e32 v1, s10, v17
	v_lshl_or_b32 v10, v10, 3, v6
	v_subrev_u32_e32 v7, s21, v18
	v_mul_lo_u32 v1, v1, s18
	v_mul_lo_u32 v8, v10, s16
	v_mul_lo_u32 v7, v7, s17
	v_or_b32_e32 v41, v2, v6
	v_sub_u32_e32 v2, v4, v5
	v_add_u32_e32 v1, v2, v1
	v_add3_u32 v19, v1, v8, v7
	v_and_b32_e32 v1, 63, v0
	v_and_b32_e32 v2, 32, v0
	v_sub_u32_e32 v1, v1, v2
	v_lshlrev_b32_e32 v34, 5, v33
	v_add_u32_e32 v52, v1, v34
	v_ashrrev_i16_e32 v4, 15, v52
	v_lshrrev_b16_e32 v4, 13, v4
	v_add_u16_e32 v4, v52, v4
	v_ashrrev_i16_e32 v5, 3, v4
	v_and_b32_e32 v4, -8, v4
	v_lshrrev_b32_e32 v2, 4, v0
	v_sub_u16_e32 v4, v52, v4
	v_and_b32_e32 v2, 2, v2
	v_bfe_i32 v53, v5, 0, 16
	v_bfe_i32 v42, v4, 0, 16
	v_mul_u32_u24_e32 v4, s37, v2
	v_mul_i32_i24_e32 v5, s31, v53
	v_lshlrev_b32_e32 v6, 3, v42
	v_add3_u32 v43, v5, v4, v6
	v_lshrrev_b32_e32 v4, 6, v0
	v_mad_i32_i24 v4, v33, -2, v4
	v_lshl_add_u32 v35, v4, 5, v1
	v_ashrrev_i32_e32 v1, 31, v35
	v_lshrrev_b32_e32 v1, 29, v1
	v_add_u32_e32 v1, v35, v1
	v_ashrrev_i32_e32 v44, 3, v1
	v_mul_lo_u32 v4, v44, s31
	v_and_b32_e32 v1, -8, v1
	s_lshl_b32 s2, s7, 1
	s_mov_b32 s3, 0x20000
	v_lshlrev_b32_e32 v9, 1, v3
	v_add_u32_e32 v10, s42, v3
	v_sub_u32_e32 v45, v35, v1
	v_mad_u32_u24 v47, v2, s37, v4
	v_lshlrev_b32_e32 v11, 1, v10
	buffer_load_dwordx4 v[1:4], v9, s[0:3], 0 offen
	buffer_load_dwordx4 v[5:8], v11, s[0:3], 0 offen
	v_add_u32_e32 v9, s42, v10
	v_lshlrev_b32_e32 v20, 1, v9
	v_add_lshl_u32 v21, v9, s42, 1
	buffer_load_dwordx4 v[9:12], v20, s[0:3], 0 offen
	buffer_load_dwordx4 v[13:16], v21, s[0:3], 0 offen
	s_sub_i32 s0, s23, s11
	v_cmp_le_i32_e32 vcc, s10, v17
	v_cmp_gt_i32_e64 s[0:1], s0, v17
	s_and_b64 s[10:11], vcc, s[0:1]
	s_sub_i32 s0, s20, s22
	v_cmp_le_i32_e32 vcc, s21, v18
	v_cmp_gt_i32_e64 s[0:1], s0, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	v_bfrev_b32_e32 v17, -2
	s_and_b64 s[0:1], s[10:11], s[0:1]
	v_cndmask_b32_e64 v25, v17, 0, s[0:1]
	s_lshl_b32 s14, s19, 1
	s_mov_b32 s15, s3
	v_lshl_add_u32 v26, v19, 1, v25
	v_add_u32_e32 v27, s16, v19
	v_lshl_add_u32 v28, v27, 1, v25
	buffer_load_dwordx4 v[17:20], v26, s[12:15], 0 offen
	buffer_load_dwordx4 v[21:24], v28, s[12:15], 0 offen
	v_add_u32_e32 v26, s16, v27
	v_lshl_add_u32 v36, v26, 1, v25
	v_add_u32_e32 v26, s16, v26
	v_lshl_add_u32 v37, v26, 1, v25
	buffer_load_dwordx4 v[25:28], v36, s[12:15], 0 offen
	buffer_load_dwordx4 v[29:32], v37, s[12:15], 0 offen
	v_add_lshl_u32 v40, v41, v40, 1
	s_movk_i32 s0, 0x4000
	s_mov_b32 s8, 0
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s12, s8
	s_mov_b32 s13, s8
	s_mov_b32 s14, s8
	s_mov_b32 s15, s8
	s_mov_b32 s16, s8
	s_mov_b32 s17, s8
	s_mov_b32 s18, s8
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	
             v_pack_b32_f16 v36, v1, v5 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v38, v1, v5, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v37, v9, v13 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v39, v9, v13, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v2, v6 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v2, v6, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v10, v14 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v10, v14, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v3, v7 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v3, v7, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v11, v15 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v11, v15, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v4, v8 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v4, v8, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v12, v16 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v12, v16, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v40, v[36:37], v[38:39] offset1:2
	ds_write2_b64 v40, v[1:2], v[5:6] offset0:4 offset1:6
	ds_write2_b64 v40, v[9:10], v[13:14] offset0:8 offset1:10
	ds_write2_b64 v40, v[3:4], v[7:8] offset0:12 offset1:14
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v17, v21 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v17, v21, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v25, v29 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v25, v29, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v17, s0, v40
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v18, v22 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v18, v22, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v26, v30 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v26, v30, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v19, v23 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v19, v23, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v27, v31 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v27, v31, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v20, v24 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v15, v20, v24, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v28, v32 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v16, v28, v32, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v17, v[1:2], v[3:4] offset0:128 offset1:130
	ds_write2_b64 v17, v[5:6], v[7:8] offset0:132 offset1:134
	ds_write2_b64 v17, v[9:10], v[11:12] offset0:136 offset1:138
	ds_write2_b64 v17, v[13:14], v[15:16] offset0:140 offset1:142
	s_mov_b32 s19, s8
	s_mov_b32 s20, s8
	s_mov_b32 s21, s8
	s_mov_b32 s22, s8
	s_mov_b32 s23, s8
	v_mov_b32_e32 v17, s8
	v_mov_b32_e32 v20, s9
	v_mov_b32_e32 v24, s10
	v_accvgpr_write_b32 a0, v17
	v_mov_b32_e32 v17, s11
	v_accvgpr_write_b32 a1, v20
	v_accvgpr_write_b32 a2, v24
	v_accvgpr_write_b32 a3, v17
	v_mov_b32_e32 v20, s12
	v_mov_b32_e32 v24, s13
	v_mov_b32_e32 v17, s14
	v_accvgpr_write_b32 a4, v20
	v_accvgpr_write_b32 a5, v24
	v_accvgpr_write_b32 a6, v17
	v_mov_b32_e32 v20, s15
	v_mov_b32_e32 v24, s16
	v_mov_b32_e32 v17, s17
	v_lshlrev_b32_e32 v9, 1, v43
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[1:4], v9 offset1:1
	v_lshlrev_b32_e32 v46, 3, v45
	v_accvgpr_write_b32 a7, v20
	v_accvgpr_write_b32 a8, v24
	v_accvgpr_write_b32 a9, v17
	v_mov_b32_e32 v20, s18
	v_mov_b32_e32 v24, s19
	v_mov_b32_e32 v17, s20
	v_add_lshl_u32 v13, v47, v46, 1
	v_add_u32_e32 v5, s0, v13
	v_accvgpr_write_b32 a10, v20
	v_accvgpr_write_b32 a11, v24
	v_accvgpr_write_b32 a12, v17
	v_mov_b32_e32 v20, s21
	v_mov_b32_e32 v24, s22
	v_mov_b32_e32 v17, s23
	ds_read2_b64 v[5:8], v5 offset0:128 offset1:129
	v_accvgpr_write_b32 a13, v20
	v_accvgpr_write_b32 a14, v24
	v_accvgpr_write_b32 a15, v17
	v_add_u32_e32 v17, 64, v35
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[5:6], a[0:15]
	s_movk_i32 s2, 0x80
	v_add_u32_e32 v25, s2, v35
	v_ashrrev_i32_e32 v18, 31, v17
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshrrev_b32_e32 v18, 29, v18
	v_lshrrev_b32_e32 v26, 29, v26
	v_add_u32_e32 v18, v17, v18
	s_mov_b32 s1, 0xffffff8
	v_add_u32_e32 v26, v25, v26
	s_movk_i32 s0, 0x1000
	v_ashrrev_i32_e32 v19, 3, v18
	v_and_b32_e32 v18, s1, v18
	v_ashrrev_i32_e32 v27, 3, v26
	v_and_b32_e32 v26, s1, v26
	s_movk_i32 s1, 0xc0
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[7:8], a[16:31]
	v_add_u32_e32 v9, s0, v9
	ds_read2_b64 v[9:12], v9 offset0:32 offset1:33
	v_add_u32_e32 v29, 0x4400, v13
	v_add_u32_e32 v13, 0x5000, v13
	ds_read2_b64 v[13:16], v13 offset0:160 offset1:161
	v_sub_u32_e32 v27, v27, v44
	v_mul_lo_u32 v27, v27, s31
	v_sub_u32_e32 v25, v25, v26
	v_sub_u32_e32 v25, v25, v45
	v_sub_u32_e32 v19, v19, v44
	v_lshl_add_u32 v25, v25, 3, v27
	v_lshl_add_u32 v30, v25, 1, v29
	v_mul_lo_u32 v19, v19, s31
	v_sub_u32_e32 v17, v17, v18
	v_sub_u32_e32 v17, v17, v45
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[16:31], v[9:10], v[13:14], a[16:31]
	v_lshl_add_u32 v17, v17, 3, v19
	v_lshl_add_u32 v21, v17, 1, v29
	ds_read2_b64 v[17:20], v21 offset1:1
	v_add_u32_e32 v21, s0, v21
	ds_read2_b64 v[21:24], v21 offset0:32 offset1:33
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_nop 7
	s_nop 3
	v_accvgpr_read_b32 v28, a16             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:68 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a17             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:72 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a18             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:76 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a19             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:80 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a20             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:84 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a21             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:88 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a22             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:92 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a23             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:96 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a24             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:100 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a25             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:104 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a26             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:108 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a27             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:112 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a28             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:116 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a29             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:120 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a30             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:124 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a31             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:128 ; 4-byte Folded Spill
	ds_read2_b64 v[25:28], v30 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[25:26], a[0:15]
	v_add_u32_e32 v30, s0, v30
	ds_read2_b64 v[36:39], v30 offset0:32 offset1:33
	v_add_u32_e32 v30, s1, v35
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshrrev_b32_e32 v31, 29, v31
	v_add_u32_e32 v31, v30, v31
	v_ashrrev_i32_e32 v32, 3, v31
	v_sub_u32_e32 v32, v32, v44
	v_mul_lo_u32 v32, v32, s31
	v_and_b32_e32 v31, 0xffffff8, v31
	v_sub_u32_e32 v30, v30, v31
	v_sub_u32_e32 v30, v30, v45
	v_lshl_add_u32 v30, v30, 3, v32
	v_lshl_add_u32 v40, v30, 1, v29
	ds_read2_b64 v[29:32], v40 offset1:1
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[27:28], a[48:63]
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[17:18], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[80:95], v[9:10], v[36:37], a[48:63]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[29:30], a[0:15]
	v_add_u32_e32 v1, s0, v40
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[19:20], a[16:31]
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[31:32], a[48:63]
	ds_read2_b64 v[1:4], v1 offset0:32 offset1:33
	v_mfma_f32_32x32x8f16 a[32:47], v[9:10], v[21:22], a[16:31]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[48:63], v[9:10], v[1:2], a[48:63]
	v_add_u32_e32 v9, 64, v52
	v_lshrrev_b32_e32 v9, 3, v9
	v_sub_u32_e32 v9, v9, v53
	v_mul_lo_u32 v9, v9, s31
	v_and_b32_e32 v10, 7, v52
	v_sub_u32_e32 v10, v10, v42
	v_lshl_add_u32 v10, v10, 3, v43
	v_add_lshl_u32 v9, v10, v9, 1
	ds_read2_b64 v[40:43], v9 offset1:1
	v_add_u32_e32 v9, s0, v9
	ds_read2_b64 v[44:47], v9 offset0:32 offset1:33
	v_add_u32_e32 v9, s2, v52
	v_lshrrev_b32_e32 v9, 3, v9
	v_sub_u32_e32 v9, v9, v53
	v_mul_lo_u32 v9, v9, s31
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[64:79], v[40:41], v[5:6], a[0:15]
	v_add_lshl_u32 v9, v10, v9, 1
	v_mfma_f32_32x32x8f16 a[96:111], v[40:41], v[25:26], a[0:15]
	v_mfma_f32_32x32x8f16 a[64:79], v[42:43], v[7:8], a[64:79]
	v_mfma_f32_32x32x8f16 a[96:111], v[42:43], v[27:28], a[96:111]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[112:127], v[44:45], v[13:14], a[64:79]
	v_mfma_f32_32x32x8f16 a[64:79], v[40:41], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[160:175], v[44:45], v[36:37], a[96:111]
	v_mfma_f32_32x32x8f16 a[96:111], v[40:41], v[29:30], a[0:15]
	v_mfma_f32_32x32x8f16 a[64:79], v[42:43], v[19:20], a[64:79]
	v_mfma_f32_32x32x8f16 a[96:111], v[42:43], v[31:32], a[96:111]
	ds_read2_b64 v[40:43], v9 offset1:1
	v_add_u32_e32 v9, s0, v9
	ds_read2_b64 v[48:51], v9 offset0:32 offset1:33
	v_add_u32_e32 v9, s1, v52
	v_lshrrev_b32_e32 v9, 3, v9
	v_sub_u32_e32 v9, v9, v53
	v_mul_lo_u32 v9, v9, s31
	v_add_lshl_u32 v9, v10, v9, 1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[128:143], v[40:41], v[5:6], a[0:15]
	v_mfma_f32_32x32x8f16 a[144:159], v[40:41], v[25:26], a[0:15]
	v_mfma_f32_32x32x8f16 a[128:143], v[42:43], v[7:8], a[128:143]
	v_mfma_f32_32x32x8f16 a[144:159], v[42:43], v[27:28], a[144:159]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[208:223], v[48:49], v[13:14], a[128:143]
	v_mfma_f32_32x32x8f16 a[128:143], v[40:41], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[48:49], v[36:37], a[144:159]
	v_mfma_f32_32x32x8f16 a[144:159], v[40:41], v[29:30], a[0:15]
	v_mfma_f32_32x32x8f16 a[128:143], v[42:43], v[19:20], a[128:143]
	v_mfma_f32_32x32x8f16 a[144:159], v[42:43], v[31:32], a[144:159]
	ds_read2_b64 v[40:43], v9 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[176:191], v[40:41], v[5:6], a[0:15]
	v_add_u32_e32 v5, s0, v9
	v_mfma_f32_32x32x8f16 a[192:207], v[40:41], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[224:239], v[40:41], v[25:26], a[0:15]
	v_mfma_f32_32x32x8f16 a[0:15], v[40:41], v[29:30], a[0:15]
	v_mfma_f32_32x32x8f16 a[192:207], v[42:43], v[19:20], a[192:207]
	v_mfma_f32_32x32x8f16 a[0:15], v[42:43], v[31:32], a[0:15]
	v_mfma_f32_32x32x8f16 a[176:191], v[42:43], v[7:8], a[176:191]
	ds_read2_b64 v[5:8], v5 offset0:32 offset1:33
	v_mfma_f32_32x32x8f16 a[64:79], v[44:45], v[21:22], a[64:79]
	v_mfma_f32_32x32x8f16 a[96:111], v[44:45], v[1:2], a[96:111]
	v_mfma_f32_32x32x8f16 a[128:143], v[48:49], v[21:22], a[128:143]
	v_mfma_f32_32x32x8f16 a[144:159], v[48:49], v[1:2], a[144:159]
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[192:207], v[5:6], v[21:22], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[42:43], v[27:28], a[224:239]
	v_mfma_f32_32x32x8f16 a[0:15], v[5:6], v[1:2], a[0:15]
	v_mfma_f32_32x32x8f16 a[16:31], v[5:6], v[36:37], a[224:239]
	v_mul_i32_i24_e32 v36, s40, v33
	v_mfma_f32_32x32x8f16 a[224:239], v[11:12], v[23:24], a[32:47]
	v_mfma_f32_32x32x8f16 a[48:63], v[11:12], v[3:4], a[48:63]
	v_mfma_f32_32x32x8f16 a[32:47], v[46:47], v[23:24], a[64:79]
	v_mfma_f32_32x32x8f16 a[64:79], v[46:47], v[3:4], a[96:111]
	v_mfma_f32_32x32x8f16 a[96:111], v[50:51], v[23:24], a[128:143]
	v_mfma_f32_32x32x8f16 a[144:159], v[50:51], v[3:4], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[7:8], v[23:24], a[192:207]
	v_mfma_f32_32x32x8f16 a[192:207], v[7:8], v[3:4], a[0:15]
	buffer_load_dword v3, off, s[56:59], 0 offset:68 ; 4-byte Folded Reload
	s_nop 7
	s_nop 7
	v_accvgpr_read_b32 v9, a128             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:4 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a129             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:8 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a130             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:12 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a131             ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[176:191], v[5:6], v[13:14], a[176:191]
	s_waitcnt vmcnt(3)
	v_accvgpr_write_b32 a0, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:72 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8f16 a[176:191], v[7:8], v[15:16], a[176:191]
	buffer_store_dword v9, off, s[56:59], 0 offset:16 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a132             ;  Reload Reuse
	s_waitcnt vmcnt(1)
	v_accvgpr_write_b32 a1, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:76 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8f16 a[160:175], v[46:47], v[38:39], a[160:175]
	buffer_store_dword v9, off, s[56:59], 0 offset:20 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a133             ;  Reload Reuse
	s_waitcnt vmcnt(1)
	v_accvgpr_write_b32 a2, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:124 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:128 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v3             ;  Reload Reuse
	s_nop 0
	v_mfma_f32_32x32x8f16 a[0:15], v[11:12], v[15:16], a[0:15]
	buffer_store_dword v9, off, s[56:59], 0 offset:24 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a134             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:28 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a135             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:32 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a136             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:36 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a137             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:40 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a138             ;  Reload Reuse
	v_accvgpr_read_b32 v32, a15
	v_accvgpr_read_b32 v31, a14
	buffer_store_dword v9, off, s[56:59], 0 offset:44 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a139             ;  Reload Reuse
	v_accvgpr_read_b32 v30, a13
	v_accvgpr_read_b32 v29, a12
	buffer_store_dword v9, off, s[56:59], 0 offset:48 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a140             ;  Reload Reuse
	v_accvgpr_read_b32 v28, a11
	v_accvgpr_read_b32 v27, a10
	buffer_store_dword v9, off, s[56:59], 0 offset:52 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a141             ;  Reload Reuse
	v_accvgpr_read_b32 v26, a9
	v_accvgpr_read_b32 v25, a8
	buffer_store_dword v9, off, s[56:59], 0 offset:56 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a142             ;  Reload Reuse
	v_accvgpr_read_b32 v24, a7
	v_accvgpr_read_b32 v23, a6
	buffer_store_dword v9, off, s[56:59], 0 offset:60 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a143             ;  Reload Reuse
	v_accvgpr_read_b32 v22, a5
	v_accvgpr_read_b32 v21, a4
	buffer_store_dword v9, off, s[56:59], 0 offset:64 ; 4-byte Folded Spill
	v_mfma_f32_32x32x8f16 a[128:143], v[46:47], v[15:16], a[112:127]
	v_accvgpr_read_b32 v20, a3
	v_accvgpr_read_b32 v19, a2
	v_accvgpr_read_b32 v18, a1
	v_accvgpr_read_b32 v17, a0
	v_mfma_f32_32x32x8f16 a[112:127], v[50:51], v[15:16], a[208:223]
	v_mfma_f32_32x32x8f16 a[208:223], v[11:12], v[38:39], a[80:95]
	v_mfma_f32_32x32x8f16 a[80:95], v[50:51], v[38:39], a[240:255]
	v_mfma_f32_32x32x8f16 a[0:15], v[7:8], v[38:39], a[16:31]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_2
; %bb.1:
	v_lshrrev_b32_e32 v1, 2, v0
	v_mul_i32_i24_e32 v2, -4, v1
	v_add_u32_e32 v1, v36, v1
	v_lshlrev_b32_e32 v3, 1, v1
	v_add_u32_e32 v4, s41, v33
	v_lshl_add_u32 v3, v4, 8, v3
	v_mul_lo_u32 v3, v3, s6
	v_add_lshl_u32 v2, v2, v0, 4
	v_lshlrev_b32_e32 v4, 12, v33
	v_lshlrev_b32_e32 v1, 7, v1
	v_add3_u32 v49, v2, v4, v1
	v_add3_u32 v48, s39, v2, v3
BB2_2:                                  ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EEC2ERSO_RKNSA_IJiiiiEEES15_S1A_RKS3_.exit.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_or_b32 v0, v0, 4, v34
	v_lshlrev_b32_e32 v33, 5, v33
	v_lshrrev_b32_e32 v34, 6, v35
	v_add3_u32 v0, v0, v36, v33
	v_sub_u32_e32 v0, v0, v34
	v_lshlrev_b32_e32 v0, 6, v0
	v_cvt_f16_f32_e32 v17, v17
	v_add_lshl_u32 v50, v0, v35, 1
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v18, v19
	v_cvt_f16_f32_e32 v19, v20
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v17
	ds_write_b16 v50, v0 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v17, v23
	v_cvt_f16_f32_e32 v18, v22
	v_cvt_f16_f32_e32 v19, v21
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v17, v26
	v_cvt_f16_f32_e32 v18, v27
	s_load_dword s2, s[4:5], 0x1b0
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v17, v31
	v_accvgpr_read_b32 v1, a224
	v_cvt_f16_f32_e32 v18, v30
	v_accvgpr_read_b32 v2, a225
	v_accvgpr_read_b32 v3, a226
	v_accvgpr_read_b32 v4, a227
	v_accvgpr_read_b32 v5, a228
	v_accvgpr_read_b32 v6, a229
	v_accvgpr_read_b32 v7, a230
	v_accvgpr_read_b32 v8, a231
	v_accvgpr_read_b32 v9, a232
	v_accvgpr_read_b32 v10, a233
	v_accvgpr_read_b32 v11, a234
	v_accvgpr_read_b32 v12, a235
	v_accvgpr_read_b32 v13, a236
	v_accvgpr_read_b32 v14, a237
	v_accvgpr_read_b32 v15, a238
	v_accvgpr_read_b32 v16, a239
	v_cvt_f16_f32_e32 v19, v29
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_4
; %bb.3:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i.i.i.i
	v_lshlrev_b32_e32 v0, 1, v49
	ds_read_b128 v[17:20], v0
	ds_read_b128 v[21:24], v0 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	buffer_store_dwordx4 v[17:20], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v17, 8, v48
	v_lshlrev_b32_e32 v18, 1, v17
	buffer_store_dwordx4 v[21:24], v18, s[24:27], 0 offen
	v_add_lshl_u32 v25, v17, s6, 1
	ds_read_b128 v[17:20], v0 offset:144
	ds_read_b128 v[21:24], v0 offset:128
	v_add_lshl_u32 v0, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[17:20], v25, s[24:27], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[21:24], v0, s[24:27], 0 offen
BB2_4:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v1, v2
	v_cvt_f16_f32_e32 v2, v3
	v_cvt_f16_f32_e32 v3, v4
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v7
	v_cvt_f16_f32_e32 v2, v6
	v_cvt_f16_f32_e32 v3, v5
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v9
	v_cvt_f16_f32_e32 v1, v10
	v_cvt_f16_f32_e32 v2, v11
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v15
	v_accvgpr_read_b32 v32, a208
	v_cvt_f16_f32_e32 v2, v14
	v_accvgpr_read_b32 v33, a209
	v_accvgpr_read_b32 v34, a210
	v_accvgpr_read_b32 v35, a211
	v_accvgpr_read_b32 v36, a212
	v_accvgpr_read_b32 v37, a213
	v_accvgpr_read_b32 v38, a214
	v_accvgpr_read_b32 v39, a215
	v_accvgpr_read_b32 v40, a216
	v_accvgpr_read_b32 v41, a217
	v_accvgpr_read_b32 v42, a218
	v_accvgpr_read_b32 v43, a219
	v_accvgpr_read_b32 v44, a220
	v_accvgpr_read_b32 v45, a221
	v_accvgpr_read_b32 v46, a222
	v_accvgpr_read_b32 v47, a223
	v_cvt_f16_f32_e32 v3, v13
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_6
; %bb.5:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i81.i.i.i.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_6:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_106.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v1, v33
	v_cvt_f16_f32_e32 v2, v34
	v_cvt_f16_f32_e32 v3, v35
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v39
	v_cvt_f16_f32_e32 v1, v38
	v_cvt_f16_f32_e32 v2, v37
	v_cvt_f16_f32_e32 v3, v36
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v40
	v_cvt_f16_f32_e32 v1, v41
	v_cvt_f16_f32_e32 v2, v42
	v_cvt_f16_f32_e32 v3, v43
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v47
	v_cvt_f16_f32_e32 v1, v46
	v_accvgpr_read_b32 v16, a48
	v_cvt_f16_f32_e32 v2, v45
	v_accvgpr_read_b32 v17, a49
	v_accvgpr_read_b32 v18, a50
	v_accvgpr_read_b32 v19, a51
	v_accvgpr_read_b32 v20, a52
	v_accvgpr_read_b32 v21, a53
	v_accvgpr_read_b32 v22, a54
	v_accvgpr_read_b32 v23, a55
	v_accvgpr_read_b32 v24, a56
	v_accvgpr_read_b32 v25, a57
	v_accvgpr_read_b32 v26, a58
	v_accvgpr_read_b32 v27, a59
	v_accvgpr_read_b32 v28, a60
	v_accvgpr_read_b32 v29, a61
	v_accvgpr_read_b32 v30, a62
	v_accvgpr_read_b32 v31, a63
	v_cvt_f16_f32_e32 v3, v44
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_8
; %bb.7:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i187.i.i.i.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_8:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_212.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a64
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a65
	v_accvgpr_read_b32 v2, a66
	v_accvgpr_read_b32 v3, a67
	v_accvgpr_read_b32 v4, a68
	v_accvgpr_read_b32 v5, a69
	v_accvgpr_read_b32 v6, a70
	v_accvgpr_read_b32 v7, a71
	v_accvgpr_read_b32 v8, a72
	v_accvgpr_read_b32 v9, a73
	v_accvgpr_read_b32 v10, a74
	v_accvgpr_read_b32 v11, a75
	v_accvgpr_read_b32 v12, a76
	v_accvgpr_read_b32 v13, a77
	v_accvgpr_read_b32 v14, a78
	v_accvgpr_read_b32 v15, a79
	v_cvt_f16_f32_e32 v19, v28
	s_mul_i32 s3, s6, 63
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_10
; %bb.9:                                ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
	v_add_lshl_u32 v25, v16, s6, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, s6, v48
	v_lshlrev_b32_e32 v17, 1, v16
	v_add_u32_e32 v48, s3, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
BB2_10:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a160
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a161
	v_accvgpr_read_b32 v18, a162
	v_accvgpr_read_b32 v19, a163
	v_accvgpr_read_b32 v20, a164
	v_accvgpr_read_b32 v21, a165
	v_accvgpr_read_b32 v22, a166
	v_accvgpr_read_b32 v23, a167
	v_accvgpr_read_b32 v24, a168
	v_accvgpr_read_b32 v25, a169
	v_accvgpr_read_b32 v26, a170
	v_accvgpr_read_b32 v27, a171
	v_accvgpr_read_b32 v28, a172
	v_accvgpr_read_b32 v29, a173
	v_accvgpr_read_b32 v30, a174
	v_accvgpr_read_b32 v31, a175
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_12
; %bb.11:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i92.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_12:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i140.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a32
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a33
	v_accvgpr_read_b32 v2, a34
	v_accvgpr_read_b32 v3, a35
	v_accvgpr_read_b32 v4, a36
	v_accvgpr_read_b32 v5, a37
	v_accvgpr_read_b32 v6, a38
	v_accvgpr_read_b32 v7, a39
	v_accvgpr_read_b32 v8, a40
	v_accvgpr_read_b32 v9, a41
	v_accvgpr_read_b32 v10, a42
	v_accvgpr_read_b32 v11, a43
	v_accvgpr_read_b32 v12, a44
	v_accvgpr_read_b32 v13, a45
	v_accvgpr_read_b32 v14, a46
	v_accvgpr_read_b32 v15, a47
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_14
; %bb.13:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i81.i.i.i192.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
	v_add_lshl_u32 v25, v16, s6, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v16, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v16, s[24:27], 0 offen
BB2_14:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_106.i.i.i240.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a128
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a129
	v_accvgpr_read_b32 v18, a130
	v_accvgpr_read_b32 v19, a131
	v_accvgpr_read_b32 v20, a132
	v_accvgpr_read_b32 v21, a133
	v_accvgpr_read_b32 v22, a134
	v_accvgpr_read_b32 v23, a135
	v_accvgpr_read_b32 v24, a136
	v_accvgpr_read_b32 v25, a137
	v_accvgpr_read_b32 v26, a138
	v_accvgpr_read_b32 v27, a139
	v_accvgpr_read_b32 v28, a140
	v_accvgpr_read_b32 v29, a141
	v_accvgpr_read_b32 v30, a142
	v_accvgpr_read_b32 v31, a143
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_16
; %bb.15:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i187.i.i.i292.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_16:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_212.i.i.i340.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a112
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a113
	v_accvgpr_read_b32 v2, a114
	v_accvgpr_read_b32 v3, a115
	v_accvgpr_read_b32 v4, a116
	v_accvgpr_read_b32 v5, a117
	v_accvgpr_read_b32 v6, a118
	v_accvgpr_read_b32 v7, a119
	v_accvgpr_read_b32 v8, a120
	v_accvgpr_read_b32 v9, a121
	v_accvgpr_read_b32 v10, a122
	v_accvgpr_read_b32 v11, a123
	v_accvgpr_read_b32 v12, a124
	v_accvgpr_read_b32 v13, a125
	v_accvgpr_read_b32 v14, a126
	v_accvgpr_read_b32 v15, a127
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_18
; %bb.17:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i380.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
	v_add_lshl_u32 v25, v16, s6, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, s6, v48
	v_lshlrev_b32_e32 v17, 1, v16
	v_add_u32_e32 v48, s3, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
BB2_18:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I405.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a96
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a97
	v_accvgpr_read_b32 v18, a98
	v_accvgpr_read_b32 v19, a99
	v_accvgpr_read_b32 v20, a100
	v_accvgpr_read_b32 v21, a101
	v_accvgpr_read_b32 v22, a102
	v_accvgpr_read_b32 v23, a103
	v_accvgpr_read_b32 v24, a104
	v_accvgpr_read_b32 v25, a105
	v_accvgpr_read_b32 v26, a106
	v_accvgpr_read_b32 v27, a107
	v_accvgpr_read_b32 v28, a108
	v_accvgpr_read_b32 v29, a109
	v_accvgpr_read_b32 v30, a110
	v_accvgpr_read_b32 v31, a111
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_20
; %bb.19:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i497.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_20:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i545.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a80
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a81
	v_accvgpr_read_b32 v2, a82
	v_accvgpr_read_b32 v3, a83
	v_accvgpr_read_b32 v4, a84
	v_accvgpr_read_b32 v5, a85
	v_accvgpr_read_b32 v6, a86
	v_accvgpr_read_b32 v7, a87
	v_accvgpr_read_b32 v8, a88
	v_accvgpr_read_b32 v9, a89
	v_accvgpr_read_b32 v10, a90
	v_accvgpr_read_b32 v11, a91
	v_accvgpr_read_b32 v12, a92
	v_accvgpr_read_b32 v13, a93
	v_accvgpr_read_b32 v14, a94
	v_accvgpr_read_b32 v15, a95
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_22
; %bb.21:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i81.i.i.i597.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
	v_add_lshl_u32 v25, v16, s6, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v16, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v16, s[24:27], 0 offen
BB2_22:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_106.i.i.i645.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a144
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a145
	v_accvgpr_read_b32 v18, a146
	v_accvgpr_read_b32 v19, a147
	v_accvgpr_read_b32 v20, a148
	v_accvgpr_read_b32 v21, a149
	v_accvgpr_read_b32 v22, a150
	v_accvgpr_read_b32 v23, a151
	v_accvgpr_read_b32 v24, a152
	v_accvgpr_read_b32 v25, a153
	v_accvgpr_read_b32 v26, a154
	v_accvgpr_read_b32 v27, a155
	v_accvgpr_read_b32 v28, a156
	v_accvgpr_read_b32 v29, a157
	v_accvgpr_read_b32 v30, a158
	v_accvgpr_read_b32 v31, a159
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_24
; %bb.23:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i187.i.i.i697.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_24:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_212.i.i.i745.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a192
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a193
	v_accvgpr_read_b32 v2, a194
	v_accvgpr_read_b32 v3, a195
	v_accvgpr_read_b32 v4, a196
	v_accvgpr_read_b32 v5, a197
	v_accvgpr_read_b32 v6, a198
	v_accvgpr_read_b32 v7, a199
	v_accvgpr_read_b32 v8, a200
	v_accvgpr_read_b32 v9, a201
	v_accvgpr_read_b32 v10, a202
	v_accvgpr_read_b32 v11, a203
	v_accvgpr_read_b32 v12, a204
	v_accvgpr_read_b32 v13, a205
	v_accvgpr_read_b32 v14, a206
	v_accvgpr_read_b32 v15, a207
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_26
; %bb.25:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i785.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
	v_add_lshl_u32 v25, v16, s6, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, s6, v48
	v_lshlrev_b32_e32 v17, 1, v16
	v_add_u32_e32 v48, s3, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
BB2_26:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I810.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v31, a15
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v30, a14
	v_accvgpr_read_b32 v29, a13
	v_accvgpr_read_b32 v28, a12
	v_accvgpr_read_b32 v27, a11
	v_accvgpr_read_b32 v26, a10
	v_accvgpr_read_b32 v25, a9
	v_accvgpr_read_b32 v24, a8
	v_accvgpr_read_b32 v23, a7
	v_accvgpr_read_b32 v22, a6
	v_accvgpr_read_b32 v21, a5
	v_accvgpr_read_b32 v20, a4
	v_accvgpr_read_b32 v19, a3
	v_accvgpr_read_b32 v18, a2
	v_accvgpr_read_b32 v17, a1
	v_accvgpr_read_b32 v16, a0
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_28
; %bb.27:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i902.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_28:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i950.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v32, off, s[56:59], 0 offset:4 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:64 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_cvt_f16_f32_e32 v18, v29
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v32            ;  Reload Reuse
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v15, a15
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_30
; %bb.29:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i81.i.i.i1002.i.i.i
	v_lshlrev_b32_e32 v24, 1, v49
	ds_read_b128 v[16:19], v24
	ds_read_b128 v[20:23], v24 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v48
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v48
	v_lshlrev_b32_e32 v17, 1, v16
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
	v_add_lshl_u32 v25, v16, s6, 1
	ds_read_b128 v[16:19], v24 offset:144
	ds_read_b128 v[20:23], v24 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v16, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v16, s[24:27], 0 offen
BB2_30:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_106.i.i.i1050.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a176
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a177
	v_accvgpr_read_b32 v18, a178
	v_accvgpr_read_b32 v19, a179
	v_accvgpr_read_b32 v20, a180
	v_accvgpr_read_b32 v21, a181
	v_accvgpr_read_b32 v22, a182
	v_accvgpr_read_b32 v23, a183
	v_accvgpr_read_b32 v24, a184
	v_accvgpr_read_b32 v25, a185
	v_accvgpr_read_b32 v26, a186
	v_accvgpr_read_b32 v27, a187
	v_accvgpr_read_b32 v28, a188
	v_accvgpr_read_b32 v29, a189
	v_accvgpr_read_b32 v30, a190
	v_accvgpr_read_b32 v31, a191
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_32
; %bb.31:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i187.i.i.i1102.i.i.i
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_32:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_212.i.i.i1150.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v17
	v_cvt_f16_f32_e32 v2, v18
	v_cvt_f16_f32_e32 v3, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v23
	v_cvt_f16_f32_e32 v1, v22
	v_cvt_f16_f32_e32 v2, v21
	v_cvt_f16_f32_e32 v3, v20
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v25
	v_cvt_f16_f32_e32 v2, v26
	v_cvt_f16_f32_e32 v3, v27
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v31
	v_cvt_f16_f32_e32 v1, v30
	v_cvt_f16_f32_e32 v2, v29
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_34
; %bb.33:
	v_lshlrev_b32_e32 v8, 1, v49
	ds_read_b128 v[0:3], v8
	ds_read_b128 v[4:7], v8 offset:16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v48
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v48
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read_b128 v[0:3], v8 offset:144
	ds_read_b128 v[4:7], v8 offset:128
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v48, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_34:                                 ; %_ZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_IJ
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
		.amdhsa_group_segment_fixed_size 34816
		.amdhsa_private_segment_fixed_size 132
		.amdhsa_kernarg_size 600
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 60
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end2:
	.size	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_, .Lfunc_end2-_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 12096
; NumSgprs: 62
; NumVgprs: 54
; NumAgprs: 256
; TotalNumVgprs: 256
; ScratchSize: 132
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 34816 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_ ; -- Begin function _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
	.globl	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
	.p2align	8
	.type	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_,@function
_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_: ; @_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
; %bb.0:
	s_mov_b64 s[58:59], s[2:3]
	s_mov_b64 s[56:57], s[0:1]
	s_add_u32 s56, s56, s7
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_load_dwordx2 s[12:13], s[4:5], 0x8
	s_load_dword s24, s[4:5], 0x48
	s_load_dword s52, s[4:5], 0x50
	s_load_dword s25, s[4:5], 0x58
	s_load_dword s53, s[4:5], 0x70
	s_load_dword s7, s[4:5], 0x84
	s_load_dwordx4 s[16:19], s[4:5], 0x98
	s_load_dwordx4 s[20:23], s[4:5], 0xac
	s_load_dwordx2 s[10:11], s[4:5], 0xbc
	s_load_dwordx2 s[2:3], s[4:5], 0xd4
	s_load_dwordx2 s[8:9], s[4:5], 0xe4
	s_load_dwordx2 s[26:27], s[4:5], 0x114
	s_load_dwordx2 s[30:31], s[4:5], 0x120
	s_load_dwordx2 s[28:29], s[4:5], 0x12c
	s_load_dwordx2 s[14:15], s[4:5], 0x13c
	s_load_dwordx2 s[36:37], s[4:5], 0x148
	s_load_dwordx2 s[34:35], s[4:5], 0x154
	s_load_dword s33, s[4:5], 0x16c
	s_waitcnt lgkmcnt(0)
	s_load_dword s19, s[4:5], 0x180
	s_load_dword s54, s[4:5], 0x1d4
	s_load_dwordx4 s[40:43], s[4:5], 0x1e0
	s_load_dwordx4 s[44:47], s[4:5], 0x1f4
	s_load_dwordx4 s[48:51], s[4:5], 0x208
	s_addc_u32 s57, s57, 0
	v_lshrrev_b32_e32 v1, 5, v0
	v_lshrrev_b32_e32 v33, 7, v0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s38, s47, s6
	s_add_i32 s38, s6, s38
	s_lshr_b32 s38, s38, s51
	s_mul_i32 s39, s38, s43
	s_sub_i32 s6, s6, s39
	s_mul_hi_u32 s39, s38, s46
	s_add_i32 s39, s38, s39
	s_lshr_b32 s43, s39, s50
	s_mul_i32 s39, s43, s42
	s_sub_i32 s39, s38, s39
	s_mul_hi_u32 s38, s43, s45
	s_add_i32 s38, s43, s38
	s_lshr_b32 s42, s38, s49
	s_mul_i32 s38, s42, s41
	s_sub_i32 s43, s43, s38
	s_mul_hi_u32 s38, s42, s44
	s_add_i32 s38, s42, s38
	s_lshr_b32 s38, s38, s48
	s_mul_i32 s40, s38, s40
	s_mul_i32 s43, s43, s54
	s_sub_i32 s41, s42, s40
	s_add_i32 s40, s6, s43
	v_mad_i32_i24 v2, v33, -4, v1
	s_mul_i32 s6, s38, s53
	v_add_u32_e32 v3, s6, v2
	v_mul_hi_u32 v4, v3, s52
	s_load_dword s42, s[4:5], 0x1c4
	s_load_dword s6, s[4:5], 0x18c
	v_lshlrev_b32_e32 v6, 2, v33
	s_mul_i32 s38, s38, s33
	v_add_u32_e32 v4, v3, v4
	v_lshrrev_b32_e32 v4, s25, v4
	v_mul_lo_u32 v5, v4, s24
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s41, s41, s42
	s_load_dwordx2 s[42:43], s[4:5], 0x24
	s_load_dwordx2 s[24:25], s[4:5], 0x10
	v_lshl_or_b32 v4, v4, 3, v6
	v_sub_u32_e32 v3, v3, v5
	s_add_i32 s41, s41, s39
	s_lshl_b32 s39, s40, 8
	s_movk_i32 s40, 0xffe0
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v4, v4, s42
	v_mul_lo_u32 v3, v3, s43
	v_mad_i32_i24 v1, v1, s40, v0
	s_lshl_b32 s44, s41, 8
	v_lshlrev_b32_e32 v5, 3, v1
	v_add_u32_e32 v7, s44, v5
	v_add3_u32 v3, v7, v4, v3
	v_add_u32_e32 v4, s39, v5
	v_mul_hi_u32 v5, v4, s37
	v_add_u32_e32 v7, s38, v2
	v_mul_hi_u32 v9, v7, s31
	s_movk_i32 s31, 0x44
	v_add_u32_e32 v5, v4, v5
	v_lshrrev_b32_e32 v5, s35, v5
	v_add_u32_e32 v9, v7, v9
	v_mul_hi_u32 v8, v5, s36
	v_lshrrev_b32_e32 v9, s29, v9
	v_mul_hi_u32 v10, v9, s30
	v_mul_lo_u32 v12, v9, s27
	v_add_u32_e32 v8, v5, v8
	v_lshrrev_b32_e32 v8, s34, v8
	v_add_u32_e32 v10, v9, v10
	v_mul_lo_u32 v11, v8, s14
	v_lshrrev_b32_e32 v10, s28, v10
	v_mul_lo_u32 v13, v10, s26
	v_mul_lo_u32 v40, v1, s31
	v_sub_u32_e32 v1, v5, v11
	v_sub_u32_e32 v7, v7, v12
	v_sub_u32_e32 v9, v9, v13
	v_mul_lo_u32 v1, v1, s8
	v_mul_lo_u32 v7, v7, s9
	v_mul_lo_u32 v8, v8, s2
	v_mul_lo_u32 v9, v9, s3
	s_movk_i32 s37, 0x880
	v_add_u32_e32 v17, v7, v1
	v_mul_lo_u32 v2, v2, s37
	v_mul_lo_u32 v5, v5, s15
	v_add_u32_e32 v18, v9, v8
	v_subrev_u32_e32 v1, s10, v17
	v_lshl_or_b32 v10, v10, 3, v6
	v_subrev_u32_e32 v7, s21, v18
	v_mul_lo_u32 v1, v1, s18
	v_mul_lo_u32 v8, v10, s16
	v_mul_lo_u32 v7, v7, s17
	v_or_b32_e32 v41, v2, v6
	v_sub_u32_e32 v2, v4, v5
	v_add_u32_e32 v1, v2, v1
	v_add3_u32 v19, v1, v8, v7
	v_and_b32_e32 v1, 63, v0
	v_and_b32_e32 v2, 32, v0
	v_sub_u32_e32 v1, v1, v2
	v_lshlrev_b32_e32 v34, 5, v33
	v_add_u32_e32 v52, v1, v34
	v_ashrrev_i16_e32 v4, 15, v52
	v_lshrrev_b16_e32 v4, 13, v4
	v_add_u16_e32 v4, v52, v4
	v_ashrrev_i16_e32 v5, 3, v4
	v_and_b32_e32 v4, -8, v4
	v_lshrrev_b32_e32 v2, 4, v0
	v_sub_u16_e32 v4, v52, v4
	v_and_b32_e32 v2, 2, v2
	v_bfe_i32 v53, v5, 0, 16
	v_bfe_i32 v42, v4, 0, 16
	v_mul_u32_u24_e32 v4, s37, v2
	v_mul_i32_i24_e32 v5, s31, v53
	v_lshlrev_b32_e32 v6, 3, v42
	v_add3_u32 v43, v5, v4, v6
	v_lshrrev_b32_e32 v4, 6, v0
	v_mad_i32_i24 v4, v33, -2, v4
	v_lshl_add_u32 v35, v4, 5, v1
	v_ashrrev_i32_e32 v1, 31, v35
	v_lshrrev_b32_e32 v1, 29, v1
	v_add_u32_e32 v1, v35, v1
	v_ashrrev_i32_e32 v44, 3, v1
	v_mul_lo_u32 v4, v44, s31
	v_and_b32_e32 v1, -8, v1
	s_lshl_b32 s2, s7, 1
	s_mov_b32 s3, 0x20000
	v_lshlrev_b32_e32 v9, 1, v3
	v_add_u32_e32 v10, s42, v3
	v_sub_u32_e32 v45, v35, v1
	v_mad_u32_u24 v47, v2, s37, v4
	v_lshlrev_b32_e32 v11, 1, v10
	buffer_load_dwordx4 v[1:4], v9, s[0:3], 0 offen
	buffer_load_dwordx4 v[5:8], v11, s[0:3], 0 offen
	v_add_u32_e32 v9, s42, v10
	v_lshlrev_b32_e32 v20, 1, v9
	v_add_lshl_u32 v21, v9, s42, 1
	buffer_load_dwordx4 v[9:12], v20, s[0:3], 0 offen
	buffer_load_dwordx4 v[13:16], v21, s[0:3], 0 offen
	s_sub_i32 s0, s23, s11
	v_cmp_le_i32_e32 vcc, s10, v17
	v_cmp_gt_i32_e64 s[0:1], s0, v17
	s_and_b64 s[10:11], vcc, s[0:1]
	s_sub_i32 s0, s20, s22
	v_cmp_le_i32_e32 vcc, s21, v18
	v_cmp_gt_i32_e64 s[0:1], s0, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	v_bfrev_b32_e32 v17, -2
	s_and_b64 s[0:1], s[10:11], s[0:1]
	v_cndmask_b32_e64 v25, v17, 0, s[0:1]
	s_lshl_b32 s14, s19, 1
	s_mov_b32 s15, s3
	v_lshl_add_u32 v26, v19, 1, v25
	v_add_u32_e32 v27, s16, v19
	v_lshl_add_u32 v28, v27, 1, v25
	buffer_load_dwordx4 v[17:20], v26, s[12:15], 0 offen
	buffer_load_dwordx4 v[21:24], v28, s[12:15], 0 offen
	v_add_u32_e32 v26, s16, v27
	v_lshl_add_u32 v36, v26, 1, v25
	v_add_u32_e32 v26, s16, v26
	v_lshl_add_u32 v37, v26, 1, v25
	buffer_load_dwordx4 v[25:28], v36, s[12:15], 0 offen
	buffer_load_dwordx4 v[29:32], v37, s[12:15], 0 offen
	v_add_lshl_u32 v40, v41, v40, 1
	s_movk_i32 s0, 0x4000
	s_mov_b32 s8, 0
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s12, s8
	s_mov_b32 s13, s8
	s_mov_b32 s14, s8
	s_mov_b32 s15, s8
	s_mov_b32 s16, s8
	s_mov_b32 s17, s8
	s_mov_b32 s18, s8
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	
             v_pack_b32_f16 v36, v1, v5 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v38, v1, v5, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v37, v9, v13 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v39, v9, v13, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v2, v6 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v2, v6, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v10, v14 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v10, v14, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v3, v7 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v3, v7, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v11, v15 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v11, v15, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v4, v8 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v4, v8, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v12, v16 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v12, v16, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v40, v[36:37], v[38:39] offset1:2
	ds_write2_b64 v40, v[1:2], v[5:6] offset0:4 offset1:6
	ds_write2_b64 v40, v[9:10], v[13:14] offset0:8 offset1:10
	ds_write2_b64 v40, v[3:4], v[7:8] offset0:12 offset1:14
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v17, v21 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v17, v21, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v25, v29 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v25, v29, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v17, s0, v40
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v18, v22 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v18, v22, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v26, v30 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v26, v30, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v19, v23 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v19, v23, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v27, v31 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v27, v31, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v20, v24 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v15, v20, v24, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v28, v32 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v16, v28, v32, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b64 v17, v[1:2], v[3:4] offset0:128 offset1:130
	ds_write2_b64 v17, v[5:6], v[7:8] offset0:132 offset1:134
	ds_write2_b64 v17, v[9:10], v[11:12] offset0:136 offset1:138
	ds_write2_b64 v17, v[13:14], v[15:16] offset0:140 offset1:142
	s_mov_b32 s19, s8
	s_mov_b32 s20, s8
	s_mov_b32 s21, s8
	s_mov_b32 s22, s8
	s_mov_b32 s23, s8
	v_mov_b32_e32 v17, s8
	v_mov_b32_e32 v20, s9
	v_mov_b32_e32 v24, s10
	v_accvgpr_write_b32 a0, v17
	v_mov_b32_e32 v17, s11
	v_accvgpr_write_b32 a1, v20
	v_accvgpr_write_b32 a2, v24
	v_accvgpr_write_b32 a3, v17
	v_mov_b32_e32 v20, s12
	v_mov_b32_e32 v24, s13
	v_mov_b32_e32 v17, s14
	v_accvgpr_write_b32 a4, v20
	v_accvgpr_write_b32 a5, v24
	v_accvgpr_write_b32 a6, v17
	v_mov_b32_e32 v20, s15
	v_mov_b32_e32 v24, s16
	v_mov_b32_e32 v17, s17
	v_lshlrev_b32_e32 v9, 1, v43
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[1:4], v9 offset1:1
	v_lshlrev_b32_e32 v46, 3, v45
	v_accvgpr_write_b32 a7, v20
	v_accvgpr_write_b32 a8, v24
	v_accvgpr_write_b32 a9, v17
	v_mov_b32_e32 v20, s18
	v_mov_b32_e32 v24, s19
	v_mov_b32_e32 v17, s20
	v_add_lshl_u32 v13, v47, v46, 1
	v_add_u32_e32 v5, s0, v13
	v_accvgpr_write_b32 a10, v20
	v_accvgpr_write_b32 a11, v24
	v_accvgpr_write_b32 a12, v17
	v_mov_b32_e32 v20, s21
	v_mov_b32_e32 v24, s22
	v_mov_b32_e32 v17, s23
	ds_read2_b64 v[5:8], v5 offset0:128 offset1:129
	v_accvgpr_write_b32 a13, v20
	v_accvgpr_write_b32 a14, v24
	v_accvgpr_write_b32 a15, v17
	v_add_u32_e32 v17, 64, v35
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[5:6], a[0:15]
	s_movk_i32 s2, 0x80
	v_add_u32_e32 v25, s2, v35
	v_ashrrev_i32_e32 v18, 31, v17
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshrrev_b32_e32 v18, 29, v18
	v_lshrrev_b32_e32 v26, 29, v26
	v_add_u32_e32 v18, v17, v18
	s_mov_b32 s1, 0xffffff8
	v_add_u32_e32 v26, v25, v26
	s_movk_i32 s0, 0x1000
	v_ashrrev_i32_e32 v19, 3, v18
	v_and_b32_e32 v18, s1, v18
	v_ashrrev_i32_e32 v27, 3, v26
	v_and_b32_e32 v26, s1, v26
	s_movk_i32 s1, 0xc0
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[7:8], a[16:31]
	v_add_u32_e32 v9, s0, v9
	ds_read2_b64 v[9:12], v9 offset0:32 offset1:33
	v_add_u32_e32 v29, 0x4400, v13
	v_add_u32_e32 v13, 0x5000, v13
	ds_read2_b64 v[13:16], v13 offset0:160 offset1:161
	v_sub_u32_e32 v27, v27, v44
	v_mul_lo_u32 v27, v27, s31
	v_sub_u32_e32 v25, v25, v26
	v_sub_u32_e32 v25, v25, v45
	v_sub_u32_e32 v19, v19, v44
	v_lshl_add_u32 v25, v25, 3, v27
	v_lshl_add_u32 v30, v25, 1, v29
	v_mul_lo_u32 v19, v19, s31
	v_sub_u32_e32 v17, v17, v18
	v_sub_u32_e32 v17, v17, v45
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[16:31], v[9:10], v[13:14], a[16:31]
	v_lshl_add_u32 v17, v17, 3, v19
	v_lshl_add_u32 v21, v17, 1, v29
	ds_read2_b64 v[17:20], v21 offset1:1
	v_add_u32_e32 v21, s0, v21
	ds_read2_b64 v[21:24], v21 offset0:32 offset1:33
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_nop 7
	s_nop 3
	v_accvgpr_read_b32 v28, a16             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:68 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a17             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:72 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a18             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:76 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a19             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:80 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a20             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:84 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a21             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:88 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a22             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:92 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a23             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:96 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a24             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:100 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a25             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:104 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a26             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:108 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a27             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:112 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a28             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:116 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a29             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:120 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a30             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:124 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v28, a31             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v28, off, s[56:59], 0 offset:128 ; 4-byte Folded Spill
	ds_read2_b64 v[25:28], v30 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[25:26], a[0:15]
	v_add_u32_e32 v30, s0, v30
	ds_read2_b64 v[36:39], v30 offset0:32 offset1:33
	v_add_u32_e32 v30, s1, v35
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshrrev_b32_e32 v31, 29, v31
	v_add_u32_e32 v31, v30, v31
	v_ashrrev_i32_e32 v32, 3, v31
	v_sub_u32_e32 v32, v32, v44
	v_mul_lo_u32 v32, v32, s31
	v_and_b32_e32 v31, 0xffffff8, v31
	v_sub_u32_e32 v30, v30, v31
	v_sub_u32_e32 v30, v30, v45
	v_lshl_add_u32 v30, v30, 3, v32
	v_lshl_add_u32 v40, v30, 1, v29
	ds_read2_b64 v[29:32], v40 offset1:1
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[27:28], a[48:63]
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[17:18], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[80:95], v[9:10], v[36:37], a[48:63]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[29:30], a[0:15]
	v_add_u32_e32 v1, s0, v40
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[19:20], a[16:31]
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[31:32], a[48:63]
	ds_read2_b64 v[1:4], v1 offset0:32 offset1:33
	v_mfma_f32_32x32x8f16 a[32:47], v[9:10], v[21:22], a[16:31]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[48:63], v[9:10], v[1:2], a[48:63]
	v_add_u32_e32 v9, 64, v52
	v_lshrrev_b32_e32 v9, 3, v9
	v_sub_u32_e32 v9, v9, v53
	v_mul_lo_u32 v9, v9, s31
	v_and_b32_e32 v10, 7, v52
	v_sub_u32_e32 v10, v10, v42
	v_lshl_add_u32 v10, v10, 3, v43
	v_add_lshl_u32 v9, v10, v9, 1
	ds_read2_b64 v[40:43], v9 offset1:1
	v_add_u32_e32 v9, s0, v9
	ds_read2_b64 v[44:47], v9 offset0:32 offset1:33
	v_add_u32_e32 v9, s2, v52
	v_lshrrev_b32_e32 v9, 3, v9
	v_sub_u32_e32 v9, v9, v53
	v_mul_lo_u32 v9, v9, s31
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[64:79], v[40:41], v[5:6], a[0:15]
	v_add_lshl_u32 v9, v10, v9, 1
	v_mfma_f32_32x32x8f16 a[96:111], v[40:41], v[25:26], a[0:15]
	v_mfma_f32_32x32x8f16 a[64:79], v[42:43], v[7:8], a[64:79]
	v_mfma_f32_32x32x8f16 a[96:111], v[42:43], v[27:28], a[96:111]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[112:127], v[44:45], v[13:14], a[64:79]
	v_mfma_f32_32x32x8f16 a[64:79], v[40:41], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[160:175], v[44:45], v[36:37], a[96:111]
	v_mfma_f32_32x32x8f16 a[96:111], v[40:41], v[29:30], a[0:15]
	v_mfma_f32_32x32x8f16 a[64:79], v[42:43], v[19:20], a[64:79]
	v_mfma_f32_32x32x8f16 a[96:111], v[42:43], v[31:32], a[96:111]
	ds_read2_b64 v[40:43], v9 offset1:1
	v_add_u32_e32 v9, s0, v9
	ds_read2_b64 v[48:51], v9 offset0:32 offset1:33
	v_add_u32_e32 v9, s1, v52
	v_lshrrev_b32_e32 v9, 3, v9
	v_sub_u32_e32 v9, v9, v53
	v_mul_lo_u32 v9, v9, s31
	v_add_lshl_u32 v9, v10, v9, 1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[128:143], v[40:41], v[5:6], a[0:15]
	v_mfma_f32_32x32x8f16 a[144:159], v[40:41], v[25:26], a[0:15]
	v_mfma_f32_32x32x8f16 a[128:143], v[42:43], v[7:8], a[128:143]
	v_mfma_f32_32x32x8f16 a[144:159], v[42:43], v[27:28], a[144:159]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[208:223], v[48:49], v[13:14], a[128:143]
	v_mfma_f32_32x32x8f16 a[128:143], v[40:41], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[240:255], v[48:49], v[36:37], a[144:159]
	v_mfma_f32_32x32x8f16 a[144:159], v[40:41], v[29:30], a[0:15]
	v_mfma_f32_32x32x8f16 a[128:143], v[42:43], v[19:20], a[128:143]
	v_mfma_f32_32x32x8f16 a[144:159], v[42:43], v[31:32], a[144:159]
	ds_read2_b64 v[40:43], v9 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[176:191], v[40:41], v[5:6], a[0:15]
	v_add_u32_e32 v5, s0, v9
	v_mfma_f32_32x32x8f16 a[192:207], v[40:41], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[224:239], v[40:41], v[25:26], a[0:15]
	v_mfma_f32_32x32x8f16 a[0:15], v[40:41], v[29:30], a[0:15]
	v_mfma_f32_32x32x8f16 a[192:207], v[42:43], v[19:20], a[192:207]
	v_mfma_f32_32x32x8f16 a[0:15], v[42:43], v[31:32], a[0:15]
	v_mfma_f32_32x32x8f16 a[176:191], v[42:43], v[7:8], a[176:191]
	ds_read2_b64 v[5:8], v5 offset0:32 offset1:33
	v_mfma_f32_32x32x8f16 a[64:79], v[44:45], v[21:22], a[64:79]
	v_mfma_f32_32x32x8f16 a[96:111], v[44:45], v[1:2], a[96:111]
	v_mfma_f32_32x32x8f16 a[128:143], v[48:49], v[21:22], a[128:143]
	v_mfma_f32_32x32x8f16 a[144:159], v[48:49], v[1:2], a[144:159]
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[192:207], v[5:6], v[21:22], a[192:207]
	v_mfma_f32_32x32x8f16 a[224:239], v[42:43], v[27:28], a[224:239]
	v_mfma_f32_32x32x8f16 a[0:15], v[5:6], v[1:2], a[0:15]
	v_mfma_f32_32x32x8f16 a[16:31], v[5:6], v[36:37], a[224:239]
	v_mul_i32_i24_e32 v36, s40, v33
	v_mfma_f32_32x32x8f16 a[224:239], v[11:12], v[23:24], a[32:47]
	v_mfma_f32_32x32x8f16 a[48:63], v[11:12], v[3:4], a[48:63]
	v_mfma_f32_32x32x8f16 a[32:47], v[46:47], v[23:24], a[64:79]
	v_mfma_f32_32x32x8f16 a[64:79], v[46:47], v[3:4], a[96:111]
	v_mfma_f32_32x32x8f16 a[96:111], v[50:51], v[23:24], a[128:143]
	v_mfma_f32_32x32x8f16 a[144:159], v[50:51], v[3:4], a[144:159]
	v_mfma_f32_32x32x8f16 a[128:143], v[7:8], v[23:24], a[192:207]
	v_mfma_f32_32x32x8f16 a[192:207], v[7:8], v[3:4], a[0:15]
	buffer_load_dword v3, off, s[56:59], 0 offset:68 ; 4-byte Folded Reload
	s_nop 7
	s_nop 7
	v_accvgpr_read_b32 v9, a128             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:4 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a129             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:8 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a130             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:12 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a131             ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[176:191], v[5:6], v[13:14], a[176:191]
	s_waitcnt vmcnt(3)
	v_accvgpr_write_b32 a0, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:72 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8f16 a[176:191], v[7:8], v[15:16], a[176:191]
	buffer_store_dword v9, off, s[56:59], 0 offset:16 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a132             ;  Reload Reuse
	s_waitcnt vmcnt(1)
	v_accvgpr_write_b32 a1, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:76 ; 4-byte Folded Reload
	v_mfma_f32_32x32x8f16 a[160:175], v[46:47], v[38:39], a[160:175]
	buffer_store_dword v9, off, s[56:59], 0 offset:20 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a133             ;  Reload Reuse
	s_waitcnt vmcnt(1)
	v_accvgpr_write_b32 a2, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v3              ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:124 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v3             ;  Reload Reuse
	buffer_load_dword v3, off, s[56:59], 0 offset:128 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v3             ;  Reload Reuse
	s_nop 0
	v_mfma_f32_32x32x8f16 a[0:15], v[11:12], v[15:16], a[0:15]
	buffer_store_dword v9, off, s[56:59], 0 offset:24 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a134             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:28 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a135             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:32 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a136             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:36 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a137             ;  Reload Reuse
	s_nop 1
	buffer_store_dword v9, off, s[56:59], 0 offset:40 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a138             ;  Reload Reuse
	v_accvgpr_read_b32 v32, a15
	v_accvgpr_read_b32 v31, a14
	buffer_store_dword v9, off, s[56:59], 0 offset:44 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a139             ;  Reload Reuse
	v_accvgpr_read_b32 v30, a13
	v_accvgpr_read_b32 v29, a12
	buffer_store_dword v9, off, s[56:59], 0 offset:48 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a140             ;  Reload Reuse
	v_accvgpr_read_b32 v28, a11
	v_accvgpr_read_b32 v27, a10
	buffer_store_dword v9, off, s[56:59], 0 offset:52 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a141             ;  Reload Reuse
	v_accvgpr_read_b32 v26, a9
	v_accvgpr_read_b32 v25, a8
	buffer_store_dword v9, off, s[56:59], 0 offset:56 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a142             ;  Reload Reuse
	v_accvgpr_read_b32 v24, a7
	v_accvgpr_read_b32 v23, a6
	buffer_store_dword v9, off, s[56:59], 0 offset:60 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v9, a143             ;  Reload Reuse
	v_accvgpr_read_b32 v22, a5
	v_accvgpr_read_b32 v21, a4
	buffer_store_dword v9, off, s[56:59], 0 offset:64 ; 4-byte Folded Spill
	v_mfma_f32_32x32x8f16 a[128:143], v[46:47], v[15:16], a[112:127]
	v_accvgpr_read_b32 v20, a3
	v_accvgpr_read_b32 v19, a2
	v_accvgpr_read_b32 v18, a1
	v_accvgpr_read_b32 v17, a0
	v_mfma_f32_32x32x8f16 a[112:127], v[50:51], v[15:16], a[208:223]
	v_mfma_f32_32x32x8f16 a[208:223], v[11:12], v[38:39], a[80:95]
	v_mfma_f32_32x32x8f16 a[80:95], v[50:51], v[38:39], a[240:255]
	v_mfma_f32_32x32x8f16 a[0:15], v[7:8], v[38:39], a[16:31]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_2
; %bb.1:
	v_lshrrev_b32_e32 v1, 2, v0
	v_mul_i32_i24_e32 v2, -4, v1
	v_add_u32_e32 v1, v36, v1
	v_lshlrev_b32_e32 v3, 1, v1
	v_add_u32_e32 v4, s41, v33
	v_lshl_add_u32 v3, v4, 8, v3
	v_mul_lo_u32 v3, v3, s6
	v_add_lshl_u32 v2, v2, v0, 4
	v_lshlrev_b32_e32 v4, 12, v33
	v_lshlrev_b32_e32 v1, 7, v1
	v_add3_u32 v49, v2, v4, v1
	v_add3_u32 v48, s39, v2, v3
BB3_2:                                  ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EEC2ERSO_RKNSA_IJiiiiEEES15_S1A_RKS3_.exit.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_or_b32 v0, v0, 4, v34
	v_lshlrev_b32_e32 v33, 5, v33
	v_lshrrev_b32_e32 v34, 6, v35
	v_add3_u32 v0, v0, v36, v33
	v_sub_u32_e32 v0, v0, v34
	v_lshlrev_b32_e32 v0, 6, v0
	v_cvt_f16_f32_e32 v17, v17
	v_add_lshl_u32 v50, v0, v35, 1
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v18, v19
	v_cvt_f16_f32_e32 v19, v20
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v17
	ds_write_b16 v50, v0 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v17, v23
	v_cvt_f16_f32_e32 v18, v22
	v_cvt_f16_f32_e32 v19, v21
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v17, v26
	v_cvt_f16_f32_e32 v18, v27
	s_load_dword s2, s[4:5], 0x1b0
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v17, v31
	v_accvgpr_read_b32 v1, a224
	v_cvt_f16_f32_e32 v18, v30
	v_accvgpr_read_b32 v2, a225
	v_accvgpr_read_b32 v3, a226
	v_accvgpr_read_b32 v4, a227
	v_accvgpr_read_b32 v5, a228
	v_accvgpr_read_b32 v6, a229
	v_accvgpr_read_b32 v7, a230
	v_accvgpr_read_b32 v8, a231
	v_accvgpr_read_b32 v9, a232
	v_accvgpr_read_b32 v10, a233
	v_accvgpr_read_b32 v11, a234
	v_accvgpr_read_b32 v12, a235
	v_accvgpr_read_b32 v13, a236
	v_accvgpr_read_b32 v14, a237
	v_accvgpr_read_b32 v15, a238
	v_accvgpr_read_b32 v16, a239
	v_cvt_f16_f32_e32 v19, v29
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_4
; %bb.3:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i.i.i.i
	v_lshlrev_b32_e32 v0, 1, v49
	ds_read2_b64 v[17:20], v0 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v20, v21, s[24:27], 12 offen
	ds_read2_b64 v[17:20], v0 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v20, v22, s[24:27], 12 offen
	ds_read2_b64 v[17:20], v0 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v20, v21, s[24:27], 12 offen
	ds_read2_b64 v[17:20], v0 offset0:16 offset1:17
	v_add_lshl_u32 v0, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v17, v0, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v18, v0, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v19, v0, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v20, v0, s[24:27], 12 offen
BB3_4:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v1, v2
	v_cvt_f16_f32_e32 v2, v3
	v_cvt_f16_f32_e32 v3, v4
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v7
	v_cvt_f16_f32_e32 v2, v6
	v_cvt_f16_f32_e32 v3, v5
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v9
	v_cvt_f16_f32_e32 v1, v10
	v_cvt_f16_f32_e32 v2, v11
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v15
	v_accvgpr_read_b32 v32, a208
	v_cvt_f16_f32_e32 v2, v14
	v_accvgpr_read_b32 v33, a209
	v_accvgpr_read_b32 v34, a210
	v_accvgpr_read_b32 v35, a211
	v_accvgpr_read_b32 v36, a212
	v_accvgpr_read_b32 v37, a213
	v_accvgpr_read_b32 v38, a214
	v_accvgpr_read_b32 v39, a215
	v_accvgpr_read_b32 v40, a216
	v_accvgpr_read_b32 v41, a217
	v_accvgpr_read_b32 v42, a218
	v_accvgpr_read_b32 v43, a219
	v_accvgpr_read_b32 v44, a220
	v_accvgpr_read_b32 v45, a221
	v_accvgpr_read_b32 v46, a222
	v_accvgpr_read_b32 v47, a223
	v_cvt_f16_f32_e32 v3, v13
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_6
; %bb.5:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i97.i.i.i.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 12 offen
BB3_6:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_122.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v1, v33
	v_cvt_f16_f32_e32 v2, v34
	v_cvt_f16_f32_e32 v3, v35
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v39
	v_cvt_f16_f32_e32 v1, v38
	v_cvt_f16_f32_e32 v2, v37
	v_cvt_f16_f32_e32 v3, v36
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v40
	v_cvt_f16_f32_e32 v1, v41
	v_cvt_f16_f32_e32 v2, v42
	v_cvt_f16_f32_e32 v3, v43
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v47
	v_cvt_f16_f32_e32 v1, v46
	v_accvgpr_read_b32 v16, a48
	v_cvt_f16_f32_e32 v2, v45
	v_accvgpr_read_b32 v17, a49
	v_accvgpr_read_b32 v18, a50
	v_accvgpr_read_b32 v19, a51
	v_accvgpr_read_b32 v20, a52
	v_accvgpr_read_b32 v21, a53
	v_accvgpr_read_b32 v22, a54
	v_accvgpr_read_b32 v23, a55
	v_accvgpr_read_b32 v24, a56
	v_accvgpr_read_b32 v25, a57
	v_accvgpr_read_b32 v26, a58
	v_accvgpr_read_b32 v27, a59
	v_accvgpr_read_b32 v28, a60
	v_accvgpr_read_b32 v29, a61
	v_accvgpr_read_b32 v30, a62
	v_accvgpr_read_b32 v31, a63
	v_cvt_f16_f32_e32 v3, v44
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_8
; %bb.7:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i219.i.i.i.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 12 offen
BB3_8:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_244.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a64
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a65
	v_accvgpr_read_b32 v2, a66
	v_accvgpr_read_b32 v3, a67
	v_accvgpr_read_b32 v4, a68
	v_accvgpr_read_b32 v5, a69
	v_accvgpr_read_b32 v6, a70
	v_accvgpr_read_b32 v7, a71
	v_accvgpr_read_b32 v8, a72
	v_accvgpr_read_b32 v9, a73
	v_accvgpr_read_b32 v10, a74
	v_accvgpr_read_b32 v11, a75
	v_accvgpr_read_b32 v12, a76
	v_accvgpr_read_b32 v13, a77
	v_accvgpr_read_b32 v14, a78
	v_accvgpr_read_b32 v15, a79
	v_cvt_f16_f32_e32 v19, v28
	s_mul_i32 s3, s6, 63
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_10
; %bb.9:                                ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_u32_e32 v21, s6, v48
	v_lshlrev_b32_e32 v20, 1, v21
	v_add_u32_e32 v48, s3, v21
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[24:27], 12 offen
BB3_10:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a160
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a161
	v_accvgpr_read_b32 v18, a162
	v_accvgpr_read_b32 v19, a163
	v_accvgpr_read_b32 v20, a164
	v_accvgpr_read_b32 v21, a165
	v_accvgpr_read_b32 v22, a166
	v_accvgpr_read_b32 v23, a167
	v_accvgpr_read_b32 v24, a168
	v_accvgpr_read_b32 v25, a169
	v_accvgpr_read_b32 v26, a170
	v_accvgpr_read_b32 v27, a171
	v_accvgpr_read_b32 v28, a172
	v_accvgpr_read_b32 v29, a173
	v_accvgpr_read_b32 v30, a174
	v_accvgpr_read_b32 v31, a175
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_12
; %bb.11:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i108.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 12 offen
BB3_12:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i156.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a32
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a33
	v_accvgpr_read_b32 v2, a34
	v_accvgpr_read_b32 v3, a35
	v_accvgpr_read_b32 v4, a36
	v_accvgpr_read_b32 v5, a37
	v_accvgpr_read_b32 v6, a38
	v_accvgpr_read_b32 v7, a39
	v_accvgpr_read_b32 v8, a40
	v_accvgpr_read_b32 v9, a41
	v_accvgpr_read_b32 v10, a42
	v_accvgpr_read_b32 v11, a43
	v_accvgpr_read_b32 v12, a44
	v_accvgpr_read_b32 v13, a45
	v_accvgpr_read_b32 v14, a46
	v_accvgpr_read_b32 v15, a47
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_14
; %bb.13:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i97.i.i.i224.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_lshl_u32 v20, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[24:27], 12 offen
BB3_14:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_122.i.i.i272.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a128
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a129
	v_accvgpr_read_b32 v18, a130
	v_accvgpr_read_b32 v19, a131
	v_accvgpr_read_b32 v20, a132
	v_accvgpr_read_b32 v21, a133
	v_accvgpr_read_b32 v22, a134
	v_accvgpr_read_b32 v23, a135
	v_accvgpr_read_b32 v24, a136
	v_accvgpr_read_b32 v25, a137
	v_accvgpr_read_b32 v26, a138
	v_accvgpr_read_b32 v27, a139
	v_accvgpr_read_b32 v28, a140
	v_accvgpr_read_b32 v29, a141
	v_accvgpr_read_b32 v30, a142
	v_accvgpr_read_b32 v31, a143
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_16
; %bb.15:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i219.i.i.i340.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 12 offen
BB3_16:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_244.i.i.i388.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a112
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a113
	v_accvgpr_read_b32 v2, a114
	v_accvgpr_read_b32 v3, a115
	v_accvgpr_read_b32 v4, a116
	v_accvgpr_read_b32 v5, a117
	v_accvgpr_read_b32 v6, a118
	v_accvgpr_read_b32 v7, a119
	v_accvgpr_read_b32 v8, a120
	v_accvgpr_read_b32 v9, a121
	v_accvgpr_read_b32 v10, a122
	v_accvgpr_read_b32 v11, a123
	v_accvgpr_read_b32 v12, a124
	v_accvgpr_read_b32 v13, a125
	v_accvgpr_read_b32 v14, a126
	v_accvgpr_read_b32 v15, a127
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_18
; %bb.17:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i444.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_u32_e32 v21, s6, v48
	v_lshlrev_b32_e32 v20, 1, v21
	v_add_u32_e32 v48, s3, v21
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[24:27], 12 offen
BB3_18:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I469.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a96
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a97
	v_accvgpr_read_b32 v18, a98
	v_accvgpr_read_b32 v19, a99
	v_accvgpr_read_b32 v20, a100
	v_accvgpr_read_b32 v21, a101
	v_accvgpr_read_b32 v22, a102
	v_accvgpr_read_b32 v23, a103
	v_accvgpr_read_b32 v24, a104
	v_accvgpr_read_b32 v25, a105
	v_accvgpr_read_b32 v26, a106
	v_accvgpr_read_b32 v27, a107
	v_accvgpr_read_b32 v28, a108
	v_accvgpr_read_b32 v29, a109
	v_accvgpr_read_b32 v30, a110
	v_accvgpr_read_b32 v31, a111
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_20
; %bb.19:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i577.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 12 offen
BB3_20:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i625.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a80
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a81
	v_accvgpr_read_b32 v2, a82
	v_accvgpr_read_b32 v3, a83
	v_accvgpr_read_b32 v4, a84
	v_accvgpr_read_b32 v5, a85
	v_accvgpr_read_b32 v6, a86
	v_accvgpr_read_b32 v7, a87
	v_accvgpr_read_b32 v8, a88
	v_accvgpr_read_b32 v9, a89
	v_accvgpr_read_b32 v10, a90
	v_accvgpr_read_b32 v11, a91
	v_accvgpr_read_b32 v12, a92
	v_accvgpr_read_b32 v13, a93
	v_accvgpr_read_b32 v14, a94
	v_accvgpr_read_b32 v15, a95
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_22
; %bb.21:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i97.i.i.i693.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_lshl_u32 v20, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[24:27], 12 offen
BB3_22:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_122.i.i.i741.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a144
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a145
	v_accvgpr_read_b32 v18, a146
	v_accvgpr_read_b32 v19, a147
	v_accvgpr_read_b32 v20, a148
	v_accvgpr_read_b32 v21, a149
	v_accvgpr_read_b32 v22, a150
	v_accvgpr_read_b32 v23, a151
	v_accvgpr_read_b32 v24, a152
	v_accvgpr_read_b32 v25, a153
	v_accvgpr_read_b32 v26, a154
	v_accvgpr_read_b32 v27, a155
	v_accvgpr_read_b32 v28, a156
	v_accvgpr_read_b32 v29, a157
	v_accvgpr_read_b32 v30, a158
	v_accvgpr_read_b32 v31, a159
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_24
; %bb.23:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i219.i.i.i809.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s6, 1
	v_add_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 12 offen
BB3_24:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_244.i.i.i857.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a192
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a193
	v_accvgpr_read_b32 v2, a194
	v_accvgpr_read_b32 v3, a195
	v_accvgpr_read_b32 v4, a196
	v_accvgpr_read_b32 v5, a197
	v_accvgpr_read_b32 v6, a198
	v_accvgpr_read_b32 v7, a199
	v_accvgpr_read_b32 v8, a200
	v_accvgpr_read_b32 v9, a201
	v_accvgpr_read_b32 v10, a202
	v_accvgpr_read_b32 v11, a203
	v_accvgpr_read_b32 v12, a204
	v_accvgpr_read_b32 v13, a205
	v_accvgpr_read_b32 v14, a206
	v_accvgpr_read_b32 v15, a207
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_26
; %bb.25:                               ; %_ZNK2ck10static_forILi0ELi4ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i913.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_u32_e32 v21, s6, v48
	v_lshlrev_b32_e32 v20, 1, v21
	v_add_u32_e32 v48, s3, v21
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[24:27], 12 offen
BB3_26:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I938.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v31, a15
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v30, a14
	v_accvgpr_read_b32 v29, a13
	v_accvgpr_read_b32 v28, a12
	v_accvgpr_read_b32 v27, a11
	v_accvgpr_read_b32 v26, a10
	v_accvgpr_read_b32 v25, a9
	v_accvgpr_read_b32 v24, a8
	v_accvgpr_read_b32 v23, a7
	v_accvgpr_read_b32 v22, a6
	v_accvgpr_read_b32 v21, a5
	v_accvgpr_read_b32 v20, a4
	v_accvgpr_read_b32 v19, a3
	v_accvgpr_read_b32 v18, a2
	v_accvgpr_read_b32 v17, a1
	v_accvgpr_read_b32 v16, a0
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_28
; %bb.27:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i.i.i.i1046.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 12 offen
BB3_28:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i1094.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v32, off, s[56:59], 0 offset:4 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:64 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v16
	ds_write_b16 v50, v17 offset:128
	ds_write_b16 v50, v18 offset:256
	ds_write_b16 v50, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v50, v16 offset:1408
	ds_write_b16 v50, v17 offset:1280
	ds_write_b16 v50, v18 offset:1152
	ds_write_b16 v50, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v50, v16 offset:2048
	ds_write_b16 v50, v17 offset:2176
	ds_write_b16 v50, v18 offset:2304
	ds_write_b16 v50, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_cvt_f16_f32_e32 v18, v29
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v50, v16 offset:3456
	ds_write_b16 v50, v17 offset:3328
	ds_write_b16 v50, v18 offset:3200
	ds_write_b16 v50, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v32            ;  Reload Reuse
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v15, a15
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_30
; %bb.29:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i97.i.i.i1162.i.i.i
	v_lshlrev_b32_e32 v20, 1, v49
	ds_read2_b64 v[16:19], v20 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v48
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:2 offset1:3
	v_add_u32_e32 v21, 8, v48
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 12 offen
	ds_read2_b64 v[16:19], v20 offset0:16 offset1:17
	v_add_lshl_u32 v20, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v16, v20, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v17, v20, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v18, v20, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v19, v20, s[24:27], 12 offen
BB3_30:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_122.i.i.i1210.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a176
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a177
	v_accvgpr_read_b32 v18, a178
	v_accvgpr_read_b32 v19, a179
	v_accvgpr_read_b32 v20, a180
	v_accvgpr_read_b32 v21, a181
	v_accvgpr_read_b32 v22, a182
	v_accvgpr_read_b32 v23, a183
	v_accvgpr_read_b32 v24, a184
	v_accvgpr_read_b32 v25, a185
	v_accvgpr_read_b32 v26, a186
	v_accvgpr_read_b32 v27, a187
	v_accvgpr_read_b32 v28, a188
	v_accvgpr_read_b32 v29, a189
	v_accvgpr_read_b32 v30, a190
	v_accvgpr_read_b32 v31, a191
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_32
; %bb.31:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEEST_EEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS18_ILS19_1EDF16_iLb1EEEEEvRSO_RKT_S15_RT0_.exit.i219.i.i.i1278.i.i.i
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s6, 1
	v_subrev_u32_e32 v48, 64, v48
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 12 offen
BB3_32:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_244.i.i.i1326.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v17
	v_cvt_f16_f32_e32 v2, v18
	v_cvt_f16_f32_e32 v3, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v50, v0
	ds_write_b16 v50, v1 offset:128
	ds_write_b16 v50, v2 offset:256
	ds_write_b16 v50, v3 offset:384
	v_cvt_f16_f32_e32 v0, v23
	v_cvt_f16_f32_e32 v1, v22
	v_cvt_f16_f32_e32 v2, v21
	v_cvt_f16_f32_e32 v3, v20
	ds_write_b16 v50, v0 offset:1408
	ds_write_b16 v50, v1 offset:1280
	ds_write_b16 v50, v2 offset:1152
	ds_write_b16 v50, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v25
	v_cvt_f16_f32_e32 v2, v26
	v_cvt_f16_f32_e32 v3, v27
	ds_write_b16 v50, v0 offset:2048
	ds_write_b16 v50, v1 offset:2176
	ds_write_b16 v50, v2 offset:2304
	ds_write_b16 v50, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v31
	v_cvt_f16_f32_e32 v1, v30
	v_cvt_f16_f32_e32 v2, v29
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v50, v0 offset:3456
	ds_write_b16 v50, v1 offset:3328
	ds_write_b16 v50, v2 offset:3200
	ds_write_b16 v50, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_34
; %bb.33:
	v_lshlrev_b32_e32 v4, 1, v49
	ds_read2_b64 v[0:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v48
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:2 offset1:3
	v_add_u32_e32 v5, 8, v48
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:18 offset1:19
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 12 offen
	ds_read2_b64 v[0:3], v4 offset0:16 offset1:17
	v_add_lshl_u32 v4, v48, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 12 offen
BB3_34:                                 ; %_ZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_IJ
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
		.amdhsa_group_segment_fixed_size 34816
		.amdhsa_private_segment_fixed_size 132
		.amdhsa_kernarg_size 600
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 60
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end3:
	.size	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_, .Lfunc_end3-_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 13572
; NumSgprs: 62
; NumVgprs: 54
; NumAgprs: 256
; TotalNumVgprs: 256
; ScratchSize: 132
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 34816 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.ident	"AMD clang version 14.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-5.0.0 22051 235b6880e2e515507478181ec11a20c1ec87945b)"
	.section	".note.GNU-stack"
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           112
        .value_kind:     by_value
      - .offset:         136
        .size:           252
        .value_kind:     by_value
      - .offset:         388
        .size:           48
        .value_kind:     by_value
      - .offset:         436
        .size:           1
        .value_kind:     by_value
      - .offset:         437
        .size:           1
        .value_kind:     by_value
      - .offset:         438
        .size:           1
        .value_kind:     by_value
      - .offset:         440
        .size:           104
        .value_kind:     by_value
      - .offset:         544
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         552
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         560
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         568
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         576
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         584
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         592
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 34816
    .kernarg_segment_align: 8
    .kernarg_segment_size: 600
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
    .private_segment_fixed_size: 68
    .sgpr_count:     70
    .sgpr_spill_count: 0
    .symbol:         _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_.kd
    .vgpr_count:     256
    .vgpr_spill_count: 16
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           112
        .value_kind:     by_value
      - .offset:         136
        .size:           252
        .value_kind:     by_value
      - .offset:         388
        .size:           48
        .value_kind:     by_value
      - .offset:         436
        .size:           1
        .value_kind:     by_value
      - .offset:         437
        .size:           1
        .value_kind:     by_value
      - .offset:         438
        .size:           1
        .value_kind:     by_value
      - .offset:         440
        .size:           104
        .value_kind:     by_value
      - .offset:         544
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         552
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         560
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         568
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         576
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         584
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         592
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 34816
    .kernarg_segment_align: 8
    .kernarg_segment_size: 600
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
    .private_segment_fixed_size: 68
    .sgpr_count:     70
    .sgpr_spill_count: 0
    .symbol:         _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_.kd
    .vgpr_count:     256
    .vgpr_spill_count: 16
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           112
        .value_kind:     by_value
      - .offset:         136
        .size:           252
        .value_kind:     by_value
      - .offset:         388
        .size:           48
        .value_kind:     by_value
      - .offset:         436
        .size:           1
        .value_kind:     by_value
      - .offset:         437
        .size:           1
        .value_kind:     by_value
      - .offset:         438
        .size:           1
        .value_kind:     by_value
      - .offset:         440
        .size:           104
        .value_kind:     by_value
      - .offset:         544
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         552
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         560
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         568
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         576
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         584
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         592
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 34816
    .kernarg_segment_align: 8
    .kernarg_segment_size: 600
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
    .private_segment_fixed_size: 132
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_.kd
    .vgpr_count:     256
    .vgpr_spill_count: 32
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           112
        .value_kind:     by_value
      - .offset:         136
        .size:           252
        .value_kind:     by_value
      - .offset:         388
        .size:           48
        .value_kind:     by_value
      - .offset:         436
        .size:           1
        .value_kind:     by_value
      - .offset:         437
        .size:           1
        .value_kind:     by_value
      - .offset:         438
        .size:           1
        .value_kind:     by_value
      - .offset:         440
        .size:           104
        .value_kind:     by_value
      - .offset:         544
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         552
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         560
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         568
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         576
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         584
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         592
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 34816
    .kernarg_segment_align: 8
    .kernarg_segment_size: 600
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_
    .private_segment_fixed_size: 132
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi256ELi4ELi32ELi32ELi8ELi4ELi4ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ES2A_S2B_S2C_Li2ELi8ELi4ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEES2H_EEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S31_PT1_T2_T3_T4_T5_T6_T7_T8_.kd
    .vgpr_count:     256
    .vgpr_spill_count: 32
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata

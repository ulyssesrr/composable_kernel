;origin loop
.origin_loop_start:
	ds_read2_b64 v_lda[0:3]
	ds_read2_b64 v_ldb[0:3]
	ds_read2_b64 v_lda[4:7]
	ds_read2_b64 v_ldb[4:7]
	v_mfma v_lda[0:1], v_ldb[0:1]
	v_mfma v_lda[2:3], v_ldb[2:3]
	v_mfma v_lda[0:1], v_ldb[4:5]
	v_mfma v_lda[2:3], v_ldb[6:7]
	v_mfma v_lda[4:5], v_ldb[0:1]
	v_mfma v_lda[6:7], v_ldb[2:3]
	v_mfma v_lda[4:5], v_ldb[4:5]
	v_mfma v_lda[6:7], v_ldb[6:7]

	ds_read2_b64 v_lda[0:3] offset: next k
	ds_read2_b64 v_lda[4:7] offset: next k
	ds_read2_b64 v_ldb[0:3] offset: next k
	ds_read2_b64 v_ldb[4:7] offset: next k

	s_barrier

	v_mfma v_lda[0:1], v_ldb[0:1]
	v_mfma v_lda[2:3], v_ldb[2:3]
	v_mfma v_lda[0:1], v_ldb[4:5]
	v_mfma v_lda[2:3], v_ldb[6:7]

	v_pack v_lda[0], v_gla[0], v_gla[1], lo
	v_pack v_lda[1], v_gla[0], v_gla[1], hi
	v_pack v_lda[2], v_gla[2], v_gla[3], lo
	v_pack v_lda[3], v_gla[2], v_gla[3], hi
	ds_write2_b64 v_lda[0:1], v_lda[2:3]

	v_pack v_pkb[0], v_glb[0], v_glb[1], lo
	v_pack v_pkb[1], v_glb[0], v_glb[1], hi
	v_pack v_pkb[2], v_glb[2], v_glb[3], lo
	v_pack v_pkb[3], v_glb[2], v_glb[3], hi
	ds_write2_b64 v_pkb[0:1], v_pkb[2:3]

	s_barrier

	v_move_slice_window 0
	v_move_slice_window 1
	; ... ~60 valus

	buffer_load_dwordx4 v_gla[0:3]
	buffer_load_dwordx4 v_glb[0:3]

	v_mfma v_lda[4:5], v_ldb[0:1]
	v_mfma v_lda[6:7], v_ldb[2:3]
	v_mfma v_lda[4:5], v_ldb[4:5]
	v_mfma v_lda[6:7], v_ldb[6:7]
	s_branch origin_loop_start


;optimized loop
.optimized_loop_start:
	ds_read2_b64 v_lda[0:3]
	ds_read2_b64 v_ldb[0:3]
	ds_read2_b64 v_lda[4:7]
	ds_read2_b64 v_ldb[4:7]
	v_mfma v_lda[0:1], v_ldb[0:1]
	v_mfma v_lda[2:3], v_ldb[2:3]
	v_mfma v_lda[0:1], v_ldb[4:5]
	v_mfma v_lda[2:3], v_ldb[6:7]
	v_mfma v_lda[4:5], v_ldb[0:1]
	v_mfma v_lda[6:7], v_ldb[2:3]
	v_mfma v_lda[4:5], v_ldb[4:5]
	v_mfma v_lda[6:7], v_ldb[6:7]

	ds_read2_b64 v_lda[8:11] offset: next k
	ds_read2_b64 v_lda[12:15] offset: next k
	ds_read2_b64 v_ldb[8:11] offset: next k
	ds_read2_b64 v_ldb[12:15] offset: next k

	v_mfma v_lda[8:9], v_ldb[8:9]
	s_barrier
	v_mfma v_lda[10:11], v_ldb[10:11]

	v_pack v_lda[0], v_gla[0], v_gla[1], lo
	v_pack v_lda[1], v_gla[0], v_gla[1], hi
	v_pack v_lda[2], v_gla[2], v_gla[3], lo
	v_pack v_lda[3], v_gla[2], v_gla[3], hi

	ds_write2_b64 v_lda[0:1], v_lda[2:3]
	v_mfma v_lda[8:9], v_ldb[12:13]

	v_pack v_pkb[0], v_glb[0], v_glb[1], lo
	v_pack v_pkb[1], v_glb[0], v_glb[1], hi
	v_pack v_pkb[2], v_glb[2], v_glb[3], lo
	v_pack v_pkb[3], v_glb[2], v_glb[3], hi
	ds_write2_b64 v_pkb[0:1], v_pkb[2:3]
	v_mfma v_lda[10:11], v_ldb[14:15]

	s_barrier
	v_mfma v_lda[12:13], v_ldb[8:9]

	v_move_slice_window 0
	v_mfma v_lda[12:13], v_ldb[10:11]
	v_move_slice_window 1

	buffer_load_dwordx4 v_gla[0:3]
	v_mfma v_lda[12:13], v_ldb[12:13]
	buffer_load_dwordx4 v_glb[0:3]
	v_mfma v_lda[14:15], v_ldb[14:15]
	s_branch optimized_loop_start

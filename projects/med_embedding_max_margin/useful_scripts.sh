.build_release/tools/extract_features.bin ./projects/med_embedding_max_margin/mednet_para_256_w0_iter_10000.caffemodel ./projects/med_embedding_max_margin/wordvec_extraction.prototxt ip2 /afs/cs.stanford.edu/u/vigneshr/scratch2/ICCV2015/data/_temp/wordvec_med_eculid_256_w0 450 CPU

.build_debug/tools/extract_features.bin ./projects/med_embedding_max_margin/mednet_para_256_test_event_test_iter_12000.caffemodel ./projects/med_embedding_max_margin/videovec_extraction.prototxt video_id_emb_test /afs/cs.stanford.edu/u/vigneshr/scratch2/ICCV2015/data/_temp/videovec_med_test_event_kit_c3_256 6 CPU


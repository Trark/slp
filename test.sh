#!/bin/sh
cargo test -p slp_shared || exit 1
cargo test -p slp_lang_htk || exit 1
cargo test -p slp_lang_hst || exit 1
cargo test -p slp_lang_hir || exit 1
cargo test -p slp_lang_cil || exit 1
cargo test -p slp_lang_cst || exit 1
cargo test -p slp_transform_preprocess || exit 1
cargo test -p slp_transform_lexer || exit 1
cargo test -p slp_transform_htk_to_hst || exit 1
cargo test -p slp_transform_hst_to_hir || exit 1
cargo test -p slp_transform_hir_to_cil || exit 1
cargo test -p slp_transform_cil_to_cst || exit 1
cargo test -p slp_transform_cst_printer || exit 1
cargo test -p slp_sequence_hlsl_to_cl || exit 1
cargo test -p slp || exit 1
(cd slipstream && cargo build) || exit 1
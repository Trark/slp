#!/bin/sh
cargo test -p slp_shared
cargo test -p slp_lang_htk
cargo test -p slp_lang_hst
cargo test -p slp_lang_hir
cargo test -p slp_lang_cil
cargo test -p slp_lang_cst
cargo test -p slp_transform_preprocess
cargo test -p slp_transform_lexer
cargo test -p slp_transform_htk_to_hst
cargo test -p slp_transform_hst_to_hir
cargo test -p slp_transform_hir_to_cil
cargo test -p slp_transform_cil_to_cst
cargo test -p slp_transform_cst_printer
cargo test -p slp_sequence_hlsl_to_cl
cargo test -p slp
cd slipstream && cargo test
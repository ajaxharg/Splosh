# Splosh Makefile — the Metal build path and the wrapper targets of IMPLEMENTATION-PLAN.md §4.8.
#
# The Metal recipe is rev4 §3.3 (proven end-to-end), with the one addition this host requires:
# the clang module cache must live inside the workspace. Without it `xcrun metal` dies with
#   unable to open output file '.../clang/ModuleCache/metal_stdlib-*.pcm': Operation not permitted
# because the Metal driver's default clang module cache is $DARWIN_USER_CACHE_DIR/clang/ModuleCache,
# outside the workspace. `CLANG_MODULE_CACHE_PATH` does NOT redirect it — that was tested and still
# fails — while `-fmodules-cache-path` does. See M0-WORK-ORDER.md M0.2 and IMPLEMENTATION-PLAN.md §4.2.

SHELL := /bin/bash
TMP      := $(CURDIR)/.build/tmp
MCACHE   := $(CURDIR)/.build/metal-module-cache
OUT      := Sources/SploshCore/Resources
METALLIB := $(OUT)/default.metallib
STD      ?= metal4.0
SDK      ?= macosx
METAL    := xcrun -sdk $(SDK) metal -std=$(STD) -fmodules-cache-path=$(MCACHE)
SHADER_SRC := Sources/Shaders/copy.metal Sources/Shaders/rmsnorm.metal Sources/Shaders/rope_mrope.metal Sources/Shaders/swiglu.metal Sources/Shaders/gemm_bf16.metal Sources/Shaders/attention_dense.metal Sources/Shaders/gdn_prepare.metal Sources/Shaders/gdn_decode.metal Sources/Shaders/gdn_gate.metal Sources/Shaders/gdn_commit.metal Sources/Shaders/gemm_q4.metal Sources/Shaders/engine.metal Sources/Shaders/engine_na.metal Sources/Shaders/engine_q8.metal Sources/Shaders/engine_gguf.metal Sources/Shaders/draft.metal $(wildcard Sources/Shaders/candidates/*.metal)
M24_TMP := $(TMP)/m2.4
M24_METALLIB := $(M24_TMP)/m2.4.metallib
M24_SHADER_SRC := Sources/Shaders/copy.metal Sources/Shaders/rmsnorm.metal Sources/Shaders/rope_mrope.metal Sources/Shaders/swiglu.metal Sources/Shaders/gemm_bf16.metal Sources/Shaders/attention_dense.metal
# M2.5 cumulative shader contract (exact exports checked by the sequential gate).
M25_SHADER_SRC := Sources/Shaders/copy.metal Sources/Shaders/rmsnorm.metal Sources/Shaders/rope_mrope.metal Sources/Shaders/swiglu.metal Sources/Shaders/gemm_bf16.metal Sources/Shaders/attention_dense.metal Sources/Shaders/gdn_prepare.metal Sources/Shaders/gdn_decode.metal Sources/Shaders/gdn_gate.metal Sources/Shaders/gdn_commit.metal
M23_TMP := $(TMP)/m2.3
M23B_TMP := $(TMP)/m2.3b
M23C_TMP := $(TMP)/m2.3c
M23D_TMP := $(TMP)/m2.3d

.PHONY: check-serve setup shaders build fixtures fixtures-verify check-m0 check-m1 check-m2 check-m2.3 check-m2.3b check-m2.3c check-m2.3d check-m2.4
setup:
	@mkdir -p $(TMP) $(MCACHE) $(OUT) inputs artifacts
	swift package --disable-sandbox resolve
	@test -f Package.resolved || { echo "setup: Package.resolved missing after resolve" >&2; exit 1; }
	@echo "setup: ok"
shaders:
	@mkdir -p $(TMP) $(MCACHE) $(OUT)
	@rm -f $(TMP)/*.air
	@for f in $(SHADER_SRC); do echo "  metal $$f"; $(METAL) -c $$f -o $(TMP)/$$(basename $$f .metal).air || exit 1; done
	xcrun -sdk $(SDK) metallib $(TMP)/*.air -o $(METALLIB)
	@echo "shaders: $(METALLIB) ($$(wc -c < $(METALLIB) | tr -d ' ') bytes)"
build:
	swift build --disable-sandbox
fixtures:
	@echo "not implemented: fixtures (M2.6d owns golden generation)" >&2; exit 69
fixtures-verify:
	@echo "not implemented: fixtures-verify (M2.6d owns golden verification)" >&2; exit 69
check-m0:
	@set -euo pipefail; \
	 echo 'M0.1 package'; \
	 swift package --disable-sandbox dump-package > /tmp/splosh-package.json; \
	 python3 tools/check_package.py /tmp/splosh-package.json; \
	 echo 'M0.2 shaders/exports'; \
	 $(MAKE) shaders SHADER_SRC=Sources/Shaders/copy.metal; \
	 test -s $(METALLIB); \
	 ./tools/check-metallib-exports $(METALLIB) copy; \
	 echo 'M0.2 build/warnings'; \
	 swift build --disable-sandbox 2>&1 | tee /tmp/m0-check-build.log; \
	 ! grep -E '^/.*/(Sources|Tests)/[^:]+:.*warning:' /tmp/m0-check-build.log; \
	 echo 'M0 filters (copy-only fixture)'; \
	 for filter in MetallibTests DeviceTests CLIArgsTests DoctorCommandTests ServerCommandTests; do ./tools/swift-test-filter $$filter; done; \
	 echo 'M0.5 CLI help/stubs'; \
	 for command in doctor serve convert bench cache oracle soak; do swift run --disable-sandbox splosh $$command --help >/tmp/m0-$$command-help.out; done; \
	 serve_pid=''; \
	 cleanup_m0_serve() { \
		 set +e; \
		 if [[ -n "$$serve_pid" ]] && kill -0 "$$serve_pid" 2>/dev/null; then kill -TERM "$$serve_pid" 2>/dev/null || true; wait "$$serve_pid" 2>/dev/null || true; fi; \
		 for _ in $$(seq 1 20); do lsof -nP -iTCP:18091 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 0.1; done; \
		 if lsof -nP -iTCP:18091 -sTCP:LISTEN >/dev/null 2>&1; then echo 'serve port 18091 remains bound' >&2; lsof -nP -iTCP:18091 -sTCP:LISTEN >&2; return 1; fi; \
		 return 0; \
	 }; \
	 trap cleanup_m0_serve EXIT; \
	 test -z "$$(lsof -nP -iTCP:18091 -sTCP:LISTEN 2>/dev/null)" || { echo 'serve port 18091 already bound' >&2; exit 1; }; \
	 swift build --disable-sandbox; \
	 serve_bin="$(CURDIR)/.build/debug/splosh"; test -x "$$serve_bin"; \
	 "$$serve_bin" serve --port 18091 >/tmp/m0-serve.out 2>/tmp/m0-serve.err & serve_pid=$$!; \
	 serve_ready=0; \
	 for _ in $$(seq 1 40); do \
		 if curl -fsS --max-time 1 http://127.0.0.1:18091/health >/dev/null 2>&1; then serve_ready=1; break; fi; \
		 kill -0 "$$serve_pid" 2>/dev/null || break; sleep 0.25; \
	 done; \
	 test "$$serve_ready" = 1 || { echo 'serve failed to become ready' >&2; cat /tmp/m0-serve.err >&2; exit 1; }; \
	 kill -TERM "$$serve_pid"; \
	 serve_status=0; wait "$$serve_pid" || serve_status=$$?; \
	 test "$$serve_status" = 0 || test "$$serve_status" = 143 || { echo "serve did not terminate cleanly (status $$serve_status)" >&2; cat /tmp/m0-serve.err >&2; exit 1; }; \
	 serve_pid=''; \
	 cleanup_m0_serve; trap - EXIT; \
	 for command in convert bench cache oracle soak; do \
		 if swift run --disable-sandbox splosh $$command >/tmp/m0-$$command.out 2>/tmp/m0-$$command.err; then \
			echo "stub $$command unexpectedly exited 0" >&2; exit 1; \
		 fi; \
		 grep -Fq "not implemented: $$command" /tmp/m0-$$command.err; \
	 done; \
	 echo 'M0.10 preflight/doctor'; \
	 ./tools/preflight.sh; \
	 swift run --disable-sandbox splosh doctor; \
	 ./tools/swift-test-filter DoctorCommandTests; \
	 echo 'M0.10 fixtures-verify expected stub failure'; \
	 if $(MAKE) fixtures-verify; then echo 'fixtures-verify unexpectedly passed' >&2; exit 1; fi; \
	 echo 'M0.8 reproducibility passes'; \
	 rm -f $(TMP)/*.air; \
	 for pass in 1 2; do \
		 $(MAKE) shaders SHADER_SRC=Sources/Shaders/copy.metal; \
		 swift build --disable-sandbox; \
		 ./tools/swift-test-filter ServerCommandTests; \
		 ./tools/check-metallib-exports $(METALLIB) copy; \
		 shasum -a 256 $(METALLIB) | awk '{print $$1}' > /tmp/m0-metallib-hash-$$pass; \
		 ./tools/check-metallib-exports $(METALLIB) copy > /tmp/m0-export-$$pass.log; \
		 grep -Fq 'exports exactly [copy]' /tmp/m0-export-$$pass.log; \
	 done; \
	 test "$$$(cat /tmp/m0-metallib-hash-1)" = "$$$(cat /tmp/m0-metallib-hash-2)"; \
	 cmp /tmp/m0-export-1.log /tmp/m0-export-2.log; \
	 echo 'M0.8 runlog evidence'; \
	 test -s audit/M0-runlog.md; \
	 test "$$(grep -c -- '- make shaders: exit 0' audit/M0-runlog.md)" -ge 2; \
	 test "$$(grep -c -- '- swift build --disable-sandbox: exit 0' audit/M0-runlog.md)" -ge 2; \
	 test "$$(grep -c -- '- ServerCommandTests: exit 0' audit/M0-runlog.md)" -ge 2; \
	 test "$$(grep -c -- '- export checker: exit 0; export set: {copy}' audit/M0-runlog.md)" -ge 2; \
	 test "$$(grep -c -- 'metallib SHA-256:' audit/M0-runlog.md)" -ge 2; \
	 echo 'M0/check-m0: PASS (M0.1-M0.10; two reproducibility passes; runlog verified)'

check-m1:
	@set -euo pipefail; \
	 echo 'M1.7 asset gate'; test -f inputs/tokenizer/tokenizer.json; test -f inputs/tokenizer/tokenizer_config.json; test -f inputs/tokenizer/chat_template.jinja; (cd inputs/tokenizer && shasum -a 256 -c SHA256SUMS); \
	 echo 'M1.9 config gate'; test -f inputs/config.json; (cd inputs && shasum -a 256 -c SHA256SUMS); python3 -c 'import json; d=json.load(open("inputs/config.json")); t=d["text_config"]; assert (t["num_hidden_layers"],t["num_key_value_heads"],t["head_dim"],t["vocab_size"])==(64,4,256,248320) and d["eos_token_id"]==[248046,248044] and d["quantization"]["bits"]==4'; \
	 for filter in TokenizerTests ChatRendererTests ToolCallParserTests ArchitectureBoundaryTests ServerCommandTests DetokenizerTests; do ./tools/swift-test-filter $$filter; done; \
	 echo 'M1.4 architecture'; python3 tools/check_architecture.py; \
	 echo 'M1.6 routes'; ./tools/m1-wire-check; echo 'M1 interoperability'; ./tools/openai-compat-check

check-m2:
	@set -o pipefail; \
	 echo 'M2 cumulative shader build'; \
	 $(MAKE) shaders; \
	 ./tools/check-metallib-exports $(METALLIB) copy rmsnorm rope_mrope swiglu gemm_bf16 attention_dense_decode attention_dense_prefill gdn_prepare gdn_decode gdn_gate gdn_commit; \
	 echo 'M2 fail-closed checker tests'; \
	 python3 tools/test_check_m2_assets.py; \
	 echo 'M2 fail-closed prerequisites and assets'; \
	 python3 tools/check_m2_assets.py; \
	 echo 'M2 ModelConfig/LayerGraph and oracle gates'; \
	 for filter in ModelConfigTests TensorInventoryTests LayerGraphTests SploshOracleTests RopeOracleTests SwigluOracleTests AttentionOracleTests GdnOracleTests GdnPrefillChunkTests; do ./tools/swift-test-filter $$filter; done; \
	 echo 'M2 converter/oracle/serve/integration gates'; \
	 swift run --disable-sandbox splosh convert --verify; \
	 swift run --disable-sandbox splosh oracle --prompt Tests/Goldens/m2_prompt.txt; \
	 swift run --disable-sandbox splosh serve --help; \
	 $(MAKE) check-model; $(MAKE) check-hardware; $(MAKE) check-integration

# Build and test the isolated M2.4 shader set without changing the cumulative M2 artifact.
# The temporary resource swap is restored by the EXIT trap, so default.metallib is never left
# claiming to be the cumulative (M2) library after this target finishes.
# Build a cumulative temporary metallib and run exactly one isolated oracle suite.
# No recipe ever writes Sources/SploshCore/Resources/default.metallib.
check-m2.3:
	@./tools/check-m23-isolated.sh m2.3 M23RmsnormTests 'copy rmsnorm' Sources/Shaders/copy.metal Sources/Shaders/rmsnorm.metal
check-m2.3b:
	@./tools/check-m23-isolated.sh m2.3b M23bRopeTests 'copy rmsnorm rope_mrope' Sources/Shaders/copy.metal Sources/Shaders/rmsnorm.metal Sources/Shaders/rope_mrope.metal
check-m2.3c:
	@./tools/check-m23-isolated.sh m2.3c M23cSwigluTests 'copy rmsnorm rope_mrope swiglu' Sources/Shaders/copy.metal Sources/Shaders/rmsnorm.metal Sources/Shaders/rope_mrope.metal Sources/Shaders/swiglu.metal
check-m2.3d:
	@./tools/check-m23-isolated.sh m2.3d M23dGemmTests 'copy rmsnorm rope_mrope swiglu gemm_bf16' Sources/Shaders/copy.metal Sources/Shaders/rmsnorm.metal Sources/Shaders/rope_mrope.metal Sources/Shaders/swiglu.metal Sources/Shaders/gemm_bf16.metal

check-m2.4:
	@set -euo pipefail; \
	 mkdir -p $(M24_TMP) $(MCACHE) $(OUT); \
	 rm -f $(M24_TMP)/*.air $(M24_METALLIB); \
	 echo 'M2.4 isolated shader build'; \
	 $(MAKE) shaders SHADER_SRC="$(M24_SHADER_SRC)" TMP="$(M24_TMP)" METALLIB="$(M24_METALLIB)"; \
	 ./tools/check-metallib-exports $(M24_METALLIB) copy rmsnorm rope_mrope swiglu gemm_bf16 attention_dense_decode attention_dense_prefill; \
	 resource_backup=$(M24_TMP)/default.metallib.before; \
	 had_resource=0; \
	 if test -e $(METALLIB); then cp $(METALLIB) "$$resource_backup"; had_resource=1; fi; \
	 restore_resource() { \
		 if test "$$had_resource" = 1; then cp "$$resource_backup" $(METALLIB); else rm -f $(METALLIB); fi; \
	 }; \
	 trap restore_resource EXIT; \
	 cp $(M24_METALLIB) $(METALLIB); \
	 echo 'M2.4 attention filtered tests'; \
	 ./tools/swift-test-filter AttentionOracleTests

# Artifact-backed end-to-end gate for the real server: reference token ids, streaming, prefix
# cache, batching and memory accounting against .build/q4/weights.splw.
check-serve: shaders
	swift build -c release --disable-sandbox
	./tools/serve-smoke

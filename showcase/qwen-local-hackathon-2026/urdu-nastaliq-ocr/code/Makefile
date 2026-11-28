# Urdu OCR. Run `make help` from urdu_ocr/.
# Windows: works from PowerShell, cmd or Git Bash with GNU make (e.g. the one in C:\Users\<you>\gcc\bin).

PY        ?= .venv/Scripts/python
PORT      ?= 8001

# llama.cpp (quantized GGUF). The CPU build leaves the GPU free for training;
# for the GPU: make llama LLAMA_BIN=E:/Models/llama.cpp-cuda NGL=99
LLAMA_BIN ?= E:/Models/llama.cpp-cpu
GGUF_DIR  ?= outputs/gguf
QUANT     ?= Q8_0
NGL       ?= 0
LLAMA_PORT ?= 8090
THREADS   ?= 8

.PHONY: help status demo demo-mock app app-local app-mock llama

# make demo: our v3 and v1 fine-tunes vs Qaari 0.1 original and Qaari + our data, side by side. The app starts its own llama-server for our
# fine-tune (port DEMO_LLAMA_PORT, stopped when the app exits). On the GPU by default; for CPU only:
#   make demo DEMO_LLAMA_BIN=E:/Models/llama.cpp-cpu DEMO_NGL=0
DEMO_MODELS     ?= ours_gguf,ours_v1_gguf,qaari,qaari_ft
DEMO_GGUF       ?= outputs/gguf/urdu-ocr-2b-v3-Q8_0.gguf
DEMO_GGUF_OLD   ?= outputs/gguf/urdu-ocr-2b-Q8_0.gguf
DEMO_LLAMA_BIN  ?= E:/Models/llama.cpp-cuda
DEMO_NGL        ?= 99
DEMO_LLAMA_PORT ?= 8092

help:  # $(info) instead of echo: identical output under cmd, PowerShell and sh
	$(info make status     Progress of the overnight training / eval chains)
	$(info make demo       Four models side by side: our v3, our v1, Qaari 0.1 original, Qaari + our data (DEMO_MODELS=... to change))
	$(info make demo-mock  The same page with fake models, for UI work)
	$(info make app        Web app with the fine-tuned model in PyTorch on the GPU (exact eval model))
	$(info make llama      Start llama-server with the quantized GGUF (QUANT=$(QUANT), CPU by default))
	$(info make app-local  Web app using that llama-server (run 'make llama' in another terminal first))
	$(info make app-mock   Web app with a fake model, for UI work)
	$(info Then open http://localhost:$(PORT))
	@cd .

status:
	$(PY) tools/status.py

demo: export UOCR_MODELS := $(DEMO_MODELS)
demo: export UOCR_LLAMA_URL := http://127.0.0.1:$(DEMO_LLAMA_PORT)
demo: export UOCR_LLAMA_BIN := $(DEMO_LLAMA_BIN)
demo: export UOCR_LLAMA_NGL := $(DEMO_NGL)
demo: export UOCR_GGUF := $(DEMO_GGUF)
demo: export UOCR_GGUF_OLD := $(DEMO_GGUF_OLD)
demo:
	$(PY) -m uvicorn app.server:app --host 127.0.0.1 --port $(PORT)

demo-mock: export UOCR_MODELS := $(DEMO_MODELS)
demo-mock: export UOCR_MOCK := 1
demo-mock:
	$(PY) -m uvicorn app.server:app --host 127.0.0.1 --port $(PORT)

app: export UOCR_MODELS := ours_torch
app:
	$(PY) -m uvicorn app.server:app --host 127.0.0.1 --port $(PORT)

app-local: export UOCR_MODELS := ours_gguf
app-local: export UOCR_LLAMA_URL := http://127.0.0.1:$(LLAMA_PORT)
app-local:
	$(PY) -m uvicorn app.server:app --host 127.0.0.1 --port $(PORT)

app-mock: export UOCR_MOCK := 1
app-mock:
	$(PY) -m uvicorn app.server:app --host 127.0.0.1 --port $(PORT)

llama:
	$(LLAMA_BIN)/llama-server -m $(GGUF_DIR)/urdu-ocr-2b-$(QUANT).gguf --mmproj $(GGUF_DIR)/urdu-ocr-2b-mmproj-f16.gguf \
		--host 127.0.0.1 --port $(LLAMA_PORT) -c 8192 -ngl $(NGL) -t $(THREADS) --jinja

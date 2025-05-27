# Model Conversion and Serving Guide

This document outlines the steps to convert Hugging Face (HF) `safetensors` models to `gguf` format with image support and serve them using the `llama-server`. Note that two `gguf` files must be generated for proper operation:

1. **Primary GGUF file** (standard model data)
2. **MMProj GGUF file** (memory-mapped projection data)

> **Important:** The conversion command must be run **twice**, once for each output file.

---

## Prerequisites

- macOS (zsh)
- Python 3.8+
- `safetensors` and other HF dependencies installed
- `llama.cpp` repository built with `llama-server` binary available

Ensure you have the conversion script and server binary in your project:
```bash
# Example paths (replace with your actual paths)
CONVERTER=/path/to/convert_hf_to_gguf.py    # e.g., path to convert_hf_to_gguf.py script
SERVER=/path/to/llama-server                # e.g., path to llama-server binary
```

---

## 1. Model Conversion

Replace the `INPUT_MODEL_DIR` and `OUT_DIR` below with your actual model paths.

### 1.1 Generate Primary GGUF File

```bash
# Convert HF safetensors to primary gguf
$CONVERTER \
  /path/to/hf_model_dir \
  --outfile /path/to/output/model-f16.gguf \
  --outtype f16
```

### 1.2 Generate MMProj GGUF File

```bash
# Convert HF safetensors to memory-mapped projection gguf
$CONVERTER \
  /path/to/hf_model_dir \
  --outfile /path/to/output/model-mmproj-f16.gguf \
  --outtype f16 \
  --mmproj
```

> You should now have two files in your output directory:
> - `Qwen2.5-VL-7B-Instruct-f16.gguf`
> - `Qwen2.5-VL-7B-Instruct--mmproj-f16.gguf`

### 1.3 Conversion Script CLI Options

Run the conversion script with `--help` to view all available flags and arguments:

```bash
python3 ./convert_hf_to_gguf.py --help
```

```text
usage: convert_hf_to_gguf.py [-h] [--vocab-only] [--outfile OUTFILE] [--outtype {f32,f16,bf16,q8_0,tq1_0,tq2_0,auto}] [--bigendian] [--use-temp-file] [--no-lazy] [--model-name MODEL_NAME] [--verbose]
                             [--split-max-tensors SPLIT_MAX_TENSORS] [--split-max-size SPLIT_MAX_SIZE] [--dry-run] [--no-tensor-first-split] [--metadata METADATA] [--print-supported-models] [--remote] [--mmproj]
                             [model]
```

- `model` (positional): Path to the directory containing the Hugging Face model.
- `-h`, `--help`: Show help message and exit.
- `--vocab-only`: Extract only the vocabulary (no weights).
- `--outfile OUTFILE`: Output file path; `{ftype}` in the default name is replaced by the `outtype`.
- `--outtype {f32,f16,bf16,q8_0,tq1_0,tq2_0,auto}`: Output data type (float32, float16, bfloat16, Q8_0, ternary Q1/Q2, or auto-select 16-bit based on input).
- `--bigendian`: Mark the output for big-endian execution environments.
- `--use-temp-file`: Use temporary files during processing (helps avoid memory limits).
- `--no-lazy`: Disable lazy evaluation; compute all tensors before writing.
- `--model-name MODEL_NAME`: Override the embedded model name.
- `--verbose`: Increase logging verbosity.
- `--split-max-tensors SPLIT_MAX_TENSORS`: Maximum number of tensors per split when sharding.
- `--split-max-size SPLIT_MAX_SIZE`: Maximum file size per split (e.g., `4G`, `500M`).
- `--dry-run`: Print the planned splits without writing any files.
- `--no-tensor-first-split`: Do not include tensors in the first split file.
- `--metadata METADATA`: Path to an authorship metadata JSON override.
- `--print-supported-models`: List all Hugging Face model IDs supported by the converter.
- `--remote`: (Experimental) Stream safetensors from the Hugging Face Hub without full download; requires `HF_TOKEN` for gated repos.
- `--mmproj`: (Experimental) Export a multimodal projection module; only for certain vision models (prefixes output with `mmproj-`).

---

## 2. Serving the Model

Start the `llama-server` with both the primary and MMProj files:

```bash
$SERVER \
  -m /path/to/output/model-f16.gguf \
  --mmproj /path/to/output/model-mmproj-f16.gguf \
  --ctx-size 30000 \
  --jinja \
  --host "0.0.0.0" \
  --port 8090 \
  --threads 8
```

- `--ctx-size`: Context window size (e.g., 30000 tokens)
- `--jinja`: Enable Jinja template support
- `--host` / `--port`: Binding interface and port
- `--threads`: Number of worker threads

---

## 3. Tips & Troubleshooting

- Verify that both `.gguf` files exist and have correct file names.
- Check Python and HF dependencies if conversion fails.
- Use `--verbose` on the conversion script for detailed logs.
- Monitor `llama-server` logs for client connection and runtime errors.

---

## 4. Useful `llama-server` Options

When starting the server, you can customize behavior with these commonly used flags:

- `-h`, `--help`             : Print usage and exit.
- `--version`                : Show version and build information.
- `-t`, `--threads N`        : Number of worker threads for generation (default: auto / -1).
- `-c`, `--ctx-size N`       : Context window size in tokens (default: loaded from model).
- `-m`, `--model FILENAME`   : Path to the primary GGUF model file.
- `--mmproj FILE`            : Path to the MMProj GGUF projection file.
- `--jinja`                  : Enable Jinja template support for chat.
- `--host HOST`              : Bind address or UNIX socket (default: `127.0.0.1`).
- `--port PORT`              : Listening port (default: `8080`).
- `--threads-http N`         : Number of threads for HTTP request handling (default: -1).
- `--timeout N`              : Read/write timeout in seconds (default: `600`).
- `--no-webui`               : Disable the built-in Web UI.
- `--metrics`                : Enable Prometheus-compatible metrics endpoint.
- `--slots`                  : Enable slots monitoring endpoint.
- `--api-key KEY`            : Require this API key for authentication.
- `--log-file FILENAME`      : Write server logs to the specified file.
- `--verbose`, `--log-verbose`: Increase logging verbosity for debugging.

---

**End of Guide**

# Anthropic API Proxy for Gemini Models 🔄

**Use Anthropic clients (like Claude Code) with Gemini backends.** 🤝

A proxy server that lets you use Anthropic clients with Gemini models via LiteLLM. 🌉


![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- Google AI Studio (Gemini) API key 🔑
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-openai.git
   cd claude-code-openai
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure Environment Variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your API keys:

   *   `ANTHROPIC_API_KEY`: (Optional) Needed only if proxying *to* Anthropic models.
   *   `GEMINI_API_KEY`: Your Google AI Studio (Gemini) API key (Required).
   *   `BIG_MODEL` (Optional): The model to map `sonnet` requests to. Defaults to `gemini-2.5-pro-preview-03-25`.
   *   `SMALL_MODEL` (Optional): The model to map `haiku` requests to. Defaults to `gemini-2.0-flash`.

   **Mapping Logic:**
   - `haiku` maps to `SMALL_MODEL` prefixed with `gemini/`.
   - `sonnet` maps to `BIG_MODEL` prefixed with `gemini/`.

4. **Run the server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```
   *(`--reload` is optional, for development)*

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use the configured Gemini models through the proxy. 🎯

## Model Mapping 🗺️

The proxy automatically maps Claude models to Gemini models:

| Claude Model | Default Mapping |
|--------------|--------------|
| haiku | gemini/gemini-2.0-flash |
| sonnet | gemini/gemini-2.5-pro-preview-03-25 |

### Supported Models

#### Gemini Models
The following Gemini models are supported with automatic `gemini/` prefix handling:
- gemini-2.5-pro-preview-03-25
- gemini-2.0-flash

### Model Prefix Handling
The proxy automatically adds the appropriate prefix to model names:
- Gemini models get the `gemini/` prefix
- The BIG_MODEL and SMALL_MODEL will get the appropriate prefix based on whether they're in the Gemini model list

For example:
- `gemini-2.5-pro-preview-03-25` becomes `gemini/gemini-2.5-pro-preview-03-25`
- Claude Sonnet will map to `gemini/gemini-2.5-pro-preview-03-25` by default

### Customizing Model Mapping

Control the mapping using environment variables in your `.env` file or directly:

```dotenv
GEMINI_API_KEY="your-google-key"
BIG_MODEL="gemini-2.5-pro-preview-03-25" # Optional, it's the default
SMALL_MODEL="gemini-2.0-flash" # Optional, it's the default
```

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Translating** the requests to Gemini format via LiteLLM 🔄
3. **Sending** the translated request to Gemini 📤
4. **Converting** the response back to Anthropic format 🔄
5. **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁
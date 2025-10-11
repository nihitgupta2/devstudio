# DevStudio MCP: AI-Powered Content Creation Server

[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Production Ready](https://img.shields.io/badge/production-ready-green.svg)](https://github.com/your-username/devstudio-mcp)
[![16 Tools](https://img.shields.io/badge/tools-16-brightgreen)](https://github.com/your-username/devstudio-mcp)
[![Multi AI Provider](https://img.shields.io/badge/AI-OpenAI%20%7C%20Claude%20%7C%20Gemini-orange)](https://github.com/your-username/devstudio-mcp)

**Production-grade MCP server for technical content creators** - Automates the entire workflow from recording to publishing with AI-powered post-production. Features **16 production-ready tools** across recording, processing, generation, and monetization.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/devstudio-mcp.git
cd devstudio-mcp

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

**Note:** FFmpeg is bundled with PyAV - no separate installation required! The package includes everything needed for professional video encoding.

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Add your AI provider API keys:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

### Running the Server

```bash
# Run the MCP server
devstudio-mcp

# Or with Python
python -m devstudio_mcp.server
```

## 🎯 Core Features

### 📹 Recording & Capture
- **Screen Recording**: H.264 video with PyAV (bundled FFmpeg - no installation required)
- **Audio Recording**: Professional audio capture with real-time processing
- **Audio/Video Muxing**: Combine separate streams into single MP4 files
- **Terminal Monitoring**: Command history and output capture
- **Screenshot Tools**: Single-shot screen capture with metadata

### 🤖 AI Processing
- **Multi-Provider Transcription**: OpenAI Whisper, Google Speech-to-Text
- **Content Analysis**: Automatic topic detection and technical term extraction
- **Code Extraction**: Smart code snippet identification and categorization
- **Chapter Detection**: Intelligent content segmentation

### 📝 Content Generation
- **Blog Posts**: Technical blog posts with embedded code
- **Documentation**: API docs, guides, and technical documentation
- **Course Outlines**: Educational content structure and curriculum
- **Summaries**: Configurable length summaries for any content

## 📊 Tools Overview

| Category | Tools | Description |
|----------|-------|-------------|
| **📹 Recording** | 6 tools | Screen recording, audio capture, screenshots, session management, multi-monitor support |
| **🤖 Processing** | 3 tools | AI transcription (Whisper), content analysis, code extraction |
| **📝 Generation** | 4 tools | Blog posts, documentation, summaries, course outlines |
| **💳 Monetization** | 3 tools | License management, feature gating, usage tracking |
| **Total** | **16 tools** | Complete content creation automation pipeline |

## 🛠️ MCP Tools Reference

DevStudio MCP provides **16 production-ready tools** across 4 categories: Recording, Processing, Generation, and Monetization.

### 📹 Recording Tools (6 tools)

#### `start_recording`
**Start a new recording session** with screen, audio, and/or terminal capture.

```json
{
    "include_screen": true,
    "include_audio": true,
    "include_terminal": false,
    "output_format": "mp4",
    "screen_id": 1,
    "auto_mux": true,
    "cleanup_source_files": false
}
```
- **Returns**: Session ID, status, output directory, recording types
- **Use case**: Begin capturing screen demos, tutorials, or presentations
- **Parameters**:
  - `auto_mux` (default: true) - Automatically combine audio+video into single MP4
  - `cleanup_source_files` (default: false) - Delete source files after muxing
- **Note**: For audio-only recording, set `include_screen=false, include_audio=true`

#### `stop_recording`
**Stop an active recording session** and retrieve output file paths.

```json
{
    "session_id": "uuid-of-session"
}
```
- **Returns**: File paths for screen/audio/terminal/combined, session duration
- **Use case**: End recording and get file locations for processing
- **Note**: When `auto_mux=true` and both screen+audio recorded, returns `combined.mp4` file

#### `capture_screen`
**Take a single screenshot** of any monitor.

```json
{
    "screen_id": 1
}
```
- **Returns**: Screenshot file path, dimensions, format, size
- **Use case**: Quick captures for documentation or bug reports

#### `list_active_sessions`
**List all active recording sessions** and their status.

- **Returns**: List of active sessions with IDs, start times, types
- **Use case**: Monitor ongoing recordings or manage multiple sessions

#### `mux_audio_video`
**Combine separate audio and video files** into a single MP4.

```json
{
    "video_path": "/path/to/video.mp4",
    "audio_path": "/path/to/audio.wav",
    "output_path": "/path/to/output.mp4"
}
```
- **Returns**: Muxed file path, input files, size
- **Use case**: Merge separately recorded streams or fix sync issues

#### `get_available_screens`
**Get information about all available monitors** and their properties.

- **Returns**: List of screens with ID, resolution, position, scale
- **Use case**: Select specific monitor for recording in multi-screen setups

---

### 🤖 Processing Tools (3 tools)

#### `transcribe_audio`
**Transcribe audio to text** using OpenAI Whisper, Google Speech-to-Text, or Anthropic.

```json
{
    "file_path": "/path/to/audio.wav",
    "provider": "openai",
    "model": "whisper-1",
    "language": "en"
}
```
- **Returns**: Text, confidence, language, duration, timestamps
- **Use case**: Convert recordings to searchable text for content creation

#### `analyze_content`
**Analyze content** for topics, technical terms, code snippets, and structure.

```json
{
    "text": "Your content text...",
    "analysis_type": "comprehensive",
    "provider": "openai"
}
```
- **Returns**: Summary, key topics, technical terms, code snippets, chapters, sentiment
- **Use case**: Auto-generate metadata, tags, and structure for content

#### `extract_code`
**Extract and categorize code snippets** from text with language detection.

```json
{
    "text": "Text with code blocks..."
}
```
- **Returns**: Code snippets with language, total count, languages found
- **Use case**: Pull code examples from transcripts for documentation

---

### 📝 Generation Tools (4 tools)

#### `generate_blog_post`
**Generate technical blog posts** from transcripts with embedded code.

```json
{
    "title": "How to Build an MCP Server",
    "transcript": "Transcript text...",
    "code_snippets": [...],
    "provider": "openai",
    "style": "technical"
}
```
- **Returns**: Formatted markdown blog post, word count, read time, metadata
- **Use case**: Turn video tutorials into blog posts automatically

#### `create_documentation`
**Generate technical documentation** (API docs, guides, tutorials).

```json
{
    "title": "API Documentation",
    "content_data": {...},
    "doc_type": "api",
    "provider": "openai"
}
```
- **Returns**: Structured documentation, word count, read time
- **Use case**: Create API docs, feature guides, or technical references

#### `generate_summary`
**Create summaries** of any content in short, medium, or long formats.

```json
{
    "text": "Content to summarize...",
    "length": "medium",
    "provider": "openai"
}
```
- **Returns**: Summary text, word count, provider info
- **Use case**: Generate descriptions for YouTube, social media, or newsletters

#### `create_course_outline`
**Generate course outlines** with modules, lessons, and learning objectives.

```json
{
    "course_title": "Advanced Python Programming",
    "learning_objectives": ["Master async/await", "Build REST APIs"],
    "duration": "4 weeks",
    "skill_level": "intermediate",
    "provider": "openai"
}
```
- **Returns**: Complete course structure, word count, read time
- **Use case**: Plan educational content or training programs

---

### 💳 Monetization Tools (3 tools)

#### `get_license_info`
**Get current license tier** and subscription information.

- **Returns**: Tier, features, usage stats, expiration, upgrade URL
- **Use case**: Check subscription status and available features

#### `check_feature_access`
**Check if a specific feature** is available in current tier.

```json
{
    "feature": "ai_transcription"
}
```
- **Returns**: Access status, current tier, upgrade URL if needed
- **Use case**: Validate feature access before executing premium operations

#### `get_usage_stats`
**Get usage statistics** and remaining quotas.

- **Returns**: Current usage, limits, tier info
- **Use case**: Monitor monthly usage and plan tier upgrades

## 🏗️ Architecture

DevStudio MCP follows production-grade MCP best practices with a clean registry pattern:

```
devstudio_mcp/
├── server.py              # Main MCP server with capability negotiation
├── registry.py            # Centralized tool/resource/prompt registry
├── config.py              # Settings and environment configuration
├── tools/                 # MCP tool implementations
│   ├── recording.py       # PyAV H.264 recording, audio capture, muxing
│   ├── processing.py      # AI transcription and analysis
│   └── generation.py      # Content generation tools
├── resources/             # MCP resource providers (Phase 2)
├── prompts/              # Reusable prompt templates (Phase 2)
└── utils/                # Utilities and error handling
    ├── exceptions.py      # MCP-compliant error handling
    └── logger.py         # Structured logging
```

## 📖 Usage Examples

### Complete Workflow Example

```python
# 1. Start recording a tutorial
session = await start_recording({
    "include_screen": true,
    "include_audio": true,
    "output_format": "mp4"
})

# 2. Stop recording and get files
result = await stop_recording({
    "session_id": session["session_id"]
})

# 3. Transcribe the audio
transcript = await transcribe_audio({
    "file_path": result["files"]["audio"],
    "provider": "openai"
})

# 4. Analyze content for structure
analysis = await analyze_content({
    "text": transcript["text"],
    "analysis_type": "comprehensive"
})

# 5. Generate blog post
blog_post = await generate_blog_post({
    "title": "Building Amazing MCP Servers",
    "transcript": transcript["text"],
    "code_snippets": analysis["code_snippets"],
    "style": "technical"
})
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for Whisper transcription | Optional |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude analysis | Optional |
| `GOOGLE_API_KEY` | Google API key for Gemini processing | Optional |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No |
| `OUTPUT_DIR` | Default output directory for recordings | No |

### Settings

```python
# devstudio_mcp/config.py
class Settings:
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    log_level: str = "INFO"
    output_dir: Path = Path("./recordings")
```

## 🧪 Testing

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=devstudio_mcp

# Test specific module
uv run pytest tests/test_recording.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) - The foundation protocol
- [FastMCP](https://github.com/jlowin/fastmcp) - Python MCP framework
- [PyAV](https://github.com/PyAV-Org/PyAV) - FFmpeg bindings with bundled binaries
- [mcpcat.io](https://mcpcat.io/) - MCP best practices and guidelines

## 📞 Support

- 📧 Email: support@devstudio.com
- 💬 Discord: [DevStudio Community](https://discord.gg/devstudio)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/devstudio-mcp/issues)

---

**Built with ❤️ for the developer community**
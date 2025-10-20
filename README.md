<div align="center">

<img src="assets/logo.png" alt="DevStudio MCP Logo" width="400"/>

# DevStudio MCP: Production-Grade Screen Recording Server

[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Production Ready](https://img.shields.io/badge/production-ready-green.svg)](https://github.com/nihitgupta2/devstudio)
[![6 Tools Active](https://img.shields.io/badge/tools-6%20active-brightgreen)](https://github.com/nihitgupta2/devstudio)
[![10 Tools in Roadmap](https://img.shields.io/badge/roadmap-10%20tools-blue)](https://github.com/nihitgupta2/devstudio)
[![AGPL License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)

**Production-grade MCP server for screen recording and demo automation** - Phase 1 release featuring **6 professional recording tools** with multi-monitor support, audio/video muxing, and seamless integration for AI-driven demo workflows.

<!-- PyPI Metadata -->
mcp-name: devstudio

</div>

## 🚀 Quick Start

### Installation

**Method 1: uvx (Recommended - Zero Install)**

```bash
# Run directly without installation
uvx devstudio-mcp
```

Perfect for AI agents and quick testing - no setup required!

**Method 2: PyPI Installation**

```bash
# Install from PyPI
pip install devstudio-mcp

# Run the server
devstudio-mcp
# Or as module: python -m devstudio_mcp.server
```

**Method 3: Development Installation**

```bash
# Clone the repository
git clone https://github.com/nihitgupta2/devstudio.git
cd devstudio

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Run the server
devstudio-mcp
```

**Note:** FFmpeg is bundled with PyAV - no separate installation required! The package includes everything needed for professional video encoding.

## 🎯 Phase 1 Features

### 📹 Professional Recording Tools (6 Active Tools)
- **Multi-Monitor Recording**: Capture any screen in multi-monitor setups
- **Audio Capture**: Professional audio recording with real-time processing
- **Audio/Video Muxing**: Automatic combination of separate streams into single MP4 files
- **Screenshot Tools**: Single-shot screen capture with metadata
- **Session Management**: Track and manage multiple recording sessions
- **Terminal Monitoring**: Command history and output capture

## 📊 Tools Overview

| Tool | Description | Status |
|------|-------------|--------|
| `start_recording` | Start screen/audio/terminal recording session | ✅ Active |
| `stop_recording` | Stop recording and get output files | ✅ Active |
| `capture_screen` | Take single screenshot | ✅ Active |
| `list_active_sessions` | List all active recording sessions | ✅ Active |
| `mux_audio_video` | Combine separate audio and video files | ✅ Active |
| `get_available_screens` | Get information about all monitors | ✅ Active |

## 🛠️ MCP Tools Reference

### 📹 Recording Tools

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
- **Features**:
  - Multi-monitor support (select specific screen with `screen_id`)
  - Auto-mux: Automatically combine audio+video into single MP4 (default: true)
  - Cleanup: Delete source files after muxing (default: false)
  - Audio-only: Set `include_screen=false, include_audio=true`

#### `stop_recording`
**Stop an active recording session** and retrieve output file paths.

```json
{
    "session_id": "uuid-of-session"
}
```
- **Returns**: File paths for screen/audio/terminal/combined, session duration
- **Use case**: End recording and get file locations
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
- **Features**: Multi-monitor support via `screen_id` parameter

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
- **Note**: Uses PyAV for professional H.264/AAC encoding

#### `get_available_screens`
**Get information about all available monitors** and their properties.

- **Returns**: List of screens with ID, resolution, position, scale
- **Use case**: Select specific monitor for recording in multi-screen setups

## 🗺️ Roadmap - Development Phases

### ✅ Phase 1: Recording Infrastructure (Current - v1.0.0)
**Status**: Production-ready
**Release**: Q4 2024

- ✅ **6 production-ready recording tools**
- ✅ Multi-monitor support with screen selection
- ✅ Audio/video muxing (H.264 + AAC)
- ✅ Professional screen capture
- ✅ Session management
- ✅ PyAV integration (bundled FFmpeg)

---

### 🚧 Phase 2: AI-Powered Processing (In Development)
**Target**: Q1 2025
**Git Branch**: `archive/phase-2-3-ai-features`

**Features:**
- 🔜 **Audio transcription** using OpenAI Whisper
- 🔜 **Multi-provider AI** (OpenAI, Anthropic, Google)
- 🔜 **Content analysis** with topic detection
- 🔜 **Code extraction** from recordings
- 🔜 **Automatic chapter detection** with timestamps

**Tools** (3):
- `transcribe_audio` - Convert audio to text with timestamps
- `analyze_content` - Extract topics, terms, and structure
- `extract_code` - Identify and categorize code snippets

---

### 📋 Phase 3: Content Generation (Planned)
**Target**: Q2 2025

**Features:**
- 📋 AI-generated **blog posts** from recordings
- 📋 Automatic **documentation creation**
- 📋 **Course outline generation**
- 📋 Multi-format content export (Markdown, HTML, PDF)
- 📋 YouTube descriptions with timestamps

**Tools** (4):
- `generate_blog_post` - Create technical blog posts
- `create_documentation` - Generate API/feature docs
- `generate_summary` - Configurable content summaries
- `create_course_outline` - Educational content structure

---

### 💎 Phase 4: Monetization & Teams (Future)
**Target**: Q3 2025

**Features:**
- 💎 **Tier-based feature access** (Free, Pro, Team, Enterprise)
- 💎 **Usage tracking and analytics**
- 💎 **License management system**
- 💎 **Team collaboration features**
- 💎 **Custom branding options**

**Tools** (3):
- `get_license_info` - Check subscription status
- `check_feature_access` - Validate feature availability
- `get_usage_stats` - Monitor usage and quotas

---

## 🔮 Vision: Autonomous Demo Recording Platform

DevStudio MCP is evolving into a **comprehensive autonomous demo recording platform** that combines professional recording infrastructure with AI-driven browser automation.

### 🌐 The Future: AI-Driven Demo Creation

**Imagine:** An AI agent that can autonomously create complete product demos by controlling browsers, recording every interaction, and generating documentation - all orchestrated through MCP.

### 🤖 AWS Nova Act Integration

**Autonomous browser control for end-to-end demo creation**

DevStudio MCP will integrate with AWS Nova Act agents to enable fully automated demo workflows:

```mermaid
graph LR
    A[MCP: Start Recording] --> B[Nova Act: Browser Automation]
    B --> C[Autonomous Actions]
    C --> D[MCP: Stop Recording]
    D --> E[Complete Demo Video]
```

#### Example Workflow: E-commerce Demo

```python
# Start recording via MCP
mcp_client.call_tool("start_recording", {
    "include_screen": true,
    "include_audio": true,
    "auto_mux": true
})

# Launch Nova Act for browser automation
from nova_act import NovaAct
nova = NovaAct(api_key="key")

# Agent performs complete workflow autonomously
nova.act("Visit amazon.com, search for wireless headphones, \
          filter by price under $100, add top rated to cart, \
          proceed to checkout page")

# Stop recording
result = mcp_client.call_tool("stop_recording", {
    "session_id": session_id
})

# Result: Professionally recorded e-commerce demo
print(f"Demo recorded: {result['files']['combined']}")
```

### 🎯 Use Cases

#### 1. **SaaS Product Demos**
Automatically record multi-step user journeys:
- Account creation and onboarding
- Feature exploration workflows
- Integration setup processes
- Admin panel operations

#### 2. **Tutorial Creation**
Capture complex technical workflows:
- Development environment setup
- API integration examples
- Debugging sessions
- Code refactoring demonstrations

#### 3. **QA & Testing**
Record test scenarios with browser interaction:
- Automated test execution capture
- Bug reproduction recordings
- Regression testing documentation
- User acceptance test (UAT) videos

#### 4. **Customer Onboarding**
Create personalized demo content:
- Product walkthroughs
- Feature tutorials
- Best practices guides
- Troubleshooting demonstrations

### 🔗 MCP Orchestration Architecture

```
┌─────────────────────────────────────────────┐
│  MCP DevStudio Recording Server (Phase 1)  │
│  • start_recording                          │
│  • capture_screen                           │
│  • stop_recording                           │
│  • multi-monitor support                    │
└──────────────┬──────────────────────────────┘
               │ Orchestrates
               ↓
┌─────────────────────────────────────────────┐
│  AWS Nova Act Browser Automation            │
│  • Navigate websites autonomously           │
│  • Fill forms with natural language         │
│  • Click & interact intelligently           │
│  • Multi-step workflow execution            │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────┐
│  AI Processing Pipeline (Phase 2-3)         │
│  • Transcribe narration                     │
│  • Generate documentation                   │
│  • Create blog posts                        │
│  • Extract code snippets                    │
└─────────────────────────────────────────────┘
               │
               ↓ Complete Demo Package
```

### 🚀 User-Defined Environment Integration

**Beyond browsers** - Record demos across any environment:

- **Local Applications**: Desktop software demos
- **Cloud Platforms**: AWS Console, Azure Portal, GCP
- **Development Tools**: VS Code, IDEs, terminals
- **Custom Systems**: Internal tools, proprietary software

Nova Act agents can be configured to interact with virtually any web-based interface, making DevStudio MCP the **universal recording infrastructure** for AI-driven demo creation.

### 🌟 Why This Matters

**Traditional demo creation:**
- ⏰ Hours of manual recording and editing
- 🎬 Multiple takes to get it right
- 📝 Manual documentation writing
- 🔄 Constant updates needed

**AI-driven with DevStudio MCP + Nova Act:**
- ⚡ **Automated end-to-end** - From browser to video
- 🎯 **Consistent quality** - Perfect execution every time
- 📚 **Auto-documentation** - AI-generated content
- 🔁 **One-click updates** - Re-record entire demos instantly

---

## 📖 Usage Examples

### Complete Recording Workflow

```python
# 1. Check available monitors
screens = await get_available_screens()
# Returns: [{"id": 0, "width": 1920, "height": 1080, ...}, ...]

# 2. Start recording a tutorial on second monitor
session = await start_recording({
    "include_screen": true,
    "include_audio": true,
    "screen_id": 1,  # Second monitor
    "output_format": "mp4",
    "auto_mux": true
})
# Returns: {"session_id": "...", "status": "recording"}

# 3. Record your demo...

# 4. Take a quick screenshot during recording
screenshot = await capture_screen({"screen_id": 1})
# Returns: {"file_path": "/path/to/screenshot.png", ...}

# 5. Stop recording and get files
result = await stop_recording({
    "session_id": session["session_id"]
})
# Returns: {"files": {"combined": "/path/to/combined.mp4"}, "duration": 120.5}
```

### Multi-Monitor Recording

```python
# Get all available screens
screens = await get_available_screens()
print(f"Found {len(screens)} monitors")

# Record specific monitor
session = await start_recording({
    "include_screen": true,
    "screen_id": 2,  # Third monitor
    "include_audio": true
})
```

### Audio/Video Muxing

```python
# Record audio and video separately
session = await start_recording({
    "include_screen": true,
    "include_audio": true,
    "auto_mux": false  # Don't auto-combine
})

result = await stop_recording({"session_id": session["session_id"]})

# Manually mux later with custom settings
muxed = await mux_audio_video({
    "video_path": result["files"]["screen"],
    "audio_path": result["files"]["audio"],
    "output_path": "/custom/path/final_demo.mp4"
})
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `RECORDING_OUTPUT_DIR` | Output directory for recordings | No | `~/devstudio/recordings` |
| `MAX_RECORDING_DURATION` | Max recording length (seconds) | No | 3600 (1 hour) |
| `RECORDING_QUALITY` | Quality setting (low, medium, high) | No | medium |
| `RECORDING_FPS` | Frames per second | No | 30 |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No | INFO |

### Default Recording Settings

```python
# Default configuration
recording:
  output_dir: ~/devstudio/recordings
  max_duration: 3600  # 1 hour
  quality: medium
  fps: 30
  audio_enabled: true
  auto_mux: true
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=devstudio_mcp

# Test recording tools specifically
uv run pytest tests/test_recording.py -v
```

## 🏗️ Architecture

```
devstudio_mcp/
├── server.py              # Main MCP server (Phase 1)
├── registry.py            # Tool registration (6 tools active)
├── config.py              # Configuration management
├── tools/
│   └── recording.py       # 6 recording tools (ACTIVE)
├── resources/
│   └── media_manager.py   # Media file management
└── utils/
    ├── exceptions.py      # Error handling
    └── logger.py         # Structured logging
```

**Archived for Phase 2/3** (preserved in git branch `archive/phase-2-3-ai-features`):
- `tools/processing.py` - AI transcription and analysis (3 tools)
- `tools/generation.py` - Content generation (4 tools)
- `monetization.py` - License management (3 tools)
- `prompts/` - AI prompt templates
- `resources/session_data.py` - Session resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing feature`)
5. Open a Pull Request

## 📄 License

DevStudio MCP is dual-licensed under **AGPL v3** (open source) and **Commercial License** (proprietary use).

### Open Source License (AGPL v3)

**Free for:**
- ✅ Personal projects and education
- ✅ Open source projects
- ✅ Internal company tools
- ✅ Research and non-commercial use

**Requirements:**
- ⚠️ If you modify this software, you must release your modifications as AGPL v3
- ⚠️ If you use this in a network service (SaaS), you must open source your entire application

Full AGPL v3 license: [LICENSE](LICENSE) | https://www.gnu.org/licenses/agpl-3.0.html

### Commercial License

**Required for:**
- 💼 Commercial SaaS offerings (proprietary)
- 💼 Selling software that includes DevStudio MCP
- 💼 Keeping your modifications proprietary
- 💼 Enterprise deployments without open sourcing

**Benefits:**
- ✅ No AGPL copyleft requirements
- ✅ Keep your code proprietary
- ✅ Priority support and updates
- ✅ Commercial use indemnification
- ✅ Custom licensing terms available

**Pricing:**
- **Startup**: < $1M revenue - Contact for pricing
- **Professional**: $1M-$10M revenue - Contact for pricing
- **Enterprise**: > $10M revenue - Custom terms

📧 **Contact for commercial licensing:** nihitgupta.ng@outlook.com

Full details: [LICENSE-COMMERCIAL.txt](LICENSE-COMMERCIAL.txt)

---

### Why Dual License?

We believe in open source while building a sustainable business. AGPL v3 ensures the community benefits from improvements, while commercial licensing allows businesses to use DevStudio MCP in proprietary applications.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) - The foundation protocol
- [FastMCP](https://github.com/jlowin/fastmcp) - Python MCP framework
- [PyAV](https://github.com/PyAV-Org/PyAV) - FFmpeg bindings with bundled binaries
- [AWS Nova Act](https://aws.amazon.com/bedrock/nova/) - Browser automation SDK
- [mcpcat.io](https://mcpcat.io/) - MCP best practices and guidelines

## 📞 Support

- 📧 Email: nihitgupta.ng@outlook.com
- 🐛 Issues: [GitHub Issues](https://github.com/nihitgupta2/devstudio/issues)
- 💼 Commercial Licensing: nihitgupta.ng@outlook.com

---

**Built with ❤️ for developers creating amazing demos and tutorials**

*Phase 2/3 features (AI processing, content generation, monetization) are actively under development and preserved in git branch: `archive/phase-2-3-ai-features`*

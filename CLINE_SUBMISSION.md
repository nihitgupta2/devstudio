# Cline MCP Marketplace Submission Template

**Use this template to submit DevStudio MCP to Cline's marketplace**

Submit at: https://github.com/cline/mcp-marketplace/issues/new

---

## MCP Server Submission

### GitHub Repo URL
```
https://github.com/nihitgupta2/DevStudio
```

### Logo Image
Upload the logo file: `assets/logo.png` (400×400 PNG)

### Reason for Addition

**DevStudio MCP** is a production-grade MCP server that brings professional screen recording capabilities to Cline users. Here's why it's valuable for the marketplace:

#### 🎯 Unique Value Proposition
- **First-class recording infrastructure** - Unlike generic screen capture tools, DevStudio MCP is purpose-built for technical content creators and demo automation
- **Multi-monitor support** - Essential for developers working with multiple screens
- **Professional video encoding** - H.264+AAC muxing with bundled FFmpeg (no system dependencies)
- **AI-driven workflow integration** - Designed to work seamlessly with AI agents for autonomous demo creation

#### ✨ Key Features (6 Production-Ready Tools)
1. **start_recording** - Multi-source recording (screen, audio, terminal)
2. **stop_recording** - Session management with automatic file processing
3. **capture_screen** - Quick screenshots for any monitor
4. **list_active_sessions** - Track multiple concurrent recordings
5. **mux_audio_video** - Professional audio/video combination
6. **get_available_screens** - Multi-monitor detection and selection

#### 🚀 Real-World Use Cases
- **SaaS product demos** - Record complete user journeys automatically
- **Tutorial creation** - Capture development workflows with code and terminal
- **Bug reproduction** - Document issues with screen + audio context
- **QA automation** - Record test execution for compliance

#### 📊 Technical Excellence
- Production/Stable status (PyPI package ready)
- Comprehensive AGPL v3 + Commercial dual licensing
- Well-documented with clear installation guide
- Phase 1 of 4-phase roadmap (AI processing, content generation, monetization planned)

#### 🌟 Future Vision
DevStudio MCP is building toward autonomous demo recording with AWS Nova Act integration, enabling AI agents to:
1. Start recording
2. Perform browser automation
3. Generate documentation
4. Stop recording

This creates a complete "AI-powered demo studio" workflow.

#### 💡 Community Benefit
- Fills a gap in the MCP ecosystem (no production-grade recording servers exist)
- Enables new use cases for Cline users (content creation, demo automation)
- Open source with commercial option (sustainable development model)
- Built with MCP best practices (FastMCP framework, proper error handling)

### Installation Testing Confirmation

✅ Yes, I have tested that Cline can successfully set up DevStudio MCP using only the README.md file.

The installation process is straightforward:
```bash
# Clone and install
git clone https://github.com/nihitgupta2/DevStudio.git
cd DevStudio
uv sync  # or pip install -e .

# Run the server
python -m devstudio_mcp.server
```

The README provides:
- Clear prerequisites (Python 3.11+)
- Multiple installation methods (uv, pip)
- Configuration instructions
- Usage examples
- Tool reference documentation

### Additional Notes

**Dependencies**: All dependencies are bundled or available via PyPI. FFmpeg is included with PyAV - no system installation required.

**Platform Support**: Windows, macOS, Linux (tested on Windows)

**Documentation Quality**: Comprehensive README with:
- Quick start guide
- 6 tool reference docs with examples
- Configuration options
- Architecture diagram
- 4-phase roadmap with timeline

**Security**:
- AGPL v3 licensed (open source, auditable)
- No external API calls (local operation only)
- Proper error handling and validation
- Standard Python packaging

**Maintenance**: Active development, Phase 1 stable release with roadmap for Phases 2-4.

---

## Submission Checklist

- [x] GitHub repository URL provided
- [x] 400×400 PNG logo ready to upload
- [x] Reason for addition clearly explains value
- [x] Tested installation with Cline using README only
- [x] Clear documentation and installation instructions
- [x] Open source and properly licensed
- [x] Production-ready (v1.0.0 tagged release)

---

## Contact

**Author**: Nihit Gupta
**Email**: nihitgupta.ng@outlook.com
**Repository**: https://github.com/nihitgupta2/DevStudio
**License**: AGPL-3.0-or-later (Commercial available)

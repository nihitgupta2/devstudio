# DevStudio MCP Installation Guide for AI Agents

This guide helps AI assistants like Claude, Cline, and other MCP clients install and configure DevStudio MCP server.

## Quick Install (Recommended)

The fastest way to use DevStudio MCP is with `uvx` (zero installation required):

```bash
uvx devstudio-mcp
```

This command automatically downloads, installs, and runs the server without any permanent installation.

## Installation Methods

### Method 1: uvx (Zero-Install, Recommended)

**Best for**: Quick testing, temporary use, AI agent automation

```bash
# Run directly without installation
uvx devstudio-mcp
```

**Configuration for MCP clients**:
```json
{
  "mcpServers": {
    "devstudio": {
      "command": "uvx",
      "args": ["devstudio-mcp"]
    }
  }
}
```

### Method 2: pip + Python Module

**Best for**: Development, customization, local projects

```bash
# Install the package
pip install devstudio-mcp

# Run as Python module
python -m devstudio_mcp.server
```

**Configuration for MCP clients**:
```json
{
  "mcpServers": {
    "devstudio": {
      "command": "python",
      "args": ["-m", "devstudio_mcp.server"]
    }
  }
}
```

### Method 3: pip + Entry Point

**Best for**: Permanent installation, system-wide access

```bash
# Install the package
pip install devstudio-mcp

# Run using command-line entry point
devstudio-mcp
```

**Configuration for MCP clients**:
```json
{
  "mcpServers": {
    "devstudio": {
      "command": "devstudio-mcp",
      "args": []
    }
  }
}
```

## Client-Specific Configuration

### Claude Desktop

Add to `claude_desktop_config.json`:

**Location**:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Configuration**:
```json
{
  "mcpServers": {
    "devstudio": {
      "command": "uvx",
      "args": ["devstudio-mcp"],
      "env": {
        "RECORDING_OUTPUT_DIR": "~/devstudio/recordings"
      }
    }
  }
}
```

### Cline (VS Code Extension)

Add to MCP settings in VS Code:

1. Open Cline settings
2. Navigate to MCP Servers configuration
3. Add DevStudio MCP:

```json
{
  "devstudio": {
    "command": "uvx",
    "args": ["devstudio-mcp"]
  }
}
```

### Cursor IDE

Add to Cursor's MCP configuration file:

```json
{
  "mcpServers": {
    "devstudio": {
      "command": "uvx",
      "args": ["devstudio-mcp"],
      "env": {
        "RECORDING_OUTPUT_DIR": "~/devstudio/recordings"
      }
    }
  }
}
```

### Zed Editor

Add to Zed's MCP configuration:

```json
{
  "context_servers": {
    "devstudio": {
      "command": "uvx",
      "args": ["devstudio-mcp"]
    }
  }
}
```

## Environment Variables (Optional)

DevStudio MCP supports the following environment variables for customization:

| Variable | Description | Default |
|----------|-------------|---------|
| `RECORDING_OUTPUT_DIR` | Output directory for recordings | `~/devstudio/recordings` |
| `MAX_RECORDING_DURATION` | Max recording length in seconds | `3600` (1 hour) |
| `RECORDING_QUALITY` | Quality setting (low, medium, high) | `medium` |
| `RECORDING_FPS` | Frames per second | `30` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

**Example with environment variables**:
```json
{
  "mcpServers": {
    "devstudio": {
      "command": "uvx",
      "args": ["devstudio-mcp"],
      "env": {
        "RECORDING_OUTPUT_DIR": "/path/to/recordings",
        "RECORDING_QUALITY": "high",
        "RECORDING_FPS": "60",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

## Verification

After installation, verify DevStudio MCP is working:

1. **Check available tools**: Your MCP client should show 6 tools:
   - `start_recording` - Start screen/audio/terminal recording
   - `stop_recording` - Stop recording and get files
   - `capture_screen` - Take single screenshot
   - `list_active_sessions` - List active recordings
   - `mux_audio_video` - Combine audio and video files
   - `get_available_screens` - Get monitor information

2. **Test basic functionality**:
   - Try: "List available screens"
   - Try: "Take a screenshot"

## Troubleshooting

### Issue: "uvx: command not found"

**Solution**: Install `uv` first:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Issue: "Module not found: devstudio_mcp"

**Solution**: Install directly with pip:
```bash
pip install devstudio-mcp
```

### Issue: FFmpeg errors

**Solution**: FFmpeg is bundled with PyAV - no separate installation required! If you still see FFmpeg errors:
```bash
# Reinstall with fresh dependencies
pip uninstall devstudio-mcp av pyav -y
pip install devstudio-mcp
```

### Issue: Permission errors on Windows

**Solution**: Run your MCP client or terminal as Administrator, or change the output directory:
```json
{
  "env": {
    "RECORDING_OUTPUT_DIR": "C:/Users/YourUsername/devstudio"
  }
}
```

### Issue: Multi-monitor recording not working

**Solution**: List available screens first:
1. Use tool: `get_available_screens`
2. Note the screen IDs
3. Pass correct `screen_id` to `start_recording`

## System Requirements

- **Python**: 3.11+ (3.10+ supported but 3.11+ recommended)
- **Operating Systems**: Windows, macOS, Linux
- **Disk Space**: ~200MB for dependencies
- **RAM**: 512MB minimum, 1GB+ recommended for video encoding

## Dependencies

DevStudio MCP includes all required dependencies:
- FastMCP (MCP framework)
- PyAV (video encoding with bundled FFmpeg)
- Pillow (image processing)
- sounddevice/soundfile (audio capture)
- screeninfo/mss (multi-monitor support)

No manual dependency installation required!

## Getting Help

- **Issues**: https://github.com/nihitgupta2/devstudio/issues
- **Email**: nihitgupta.ng@outlook.com
- **Documentation**: https://github.com/nihitgupta2/devstudio#readme

## Quick Start Example

After installation, try this workflow:

```
1. "List available screens" → See your monitors
2. "Start recording with screen and audio" → Begin recording
3. [Record your content]
4. "Stop the recording" → Get video file path
5. "Take a screenshot" → Capture current screen
```

## License

DevStudio MCP is dual-licensed:
- **AGPL-3.0** for open source use
- **Commercial License** for proprietary applications

Contact nihitgupta.ng@outlook.com for commercial licensing.

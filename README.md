# Concept Explorer

Take a few concepts, generate a few properties, add a few cognitive moves and create ideas.

A web application that lets you "mash" concepts together and visualize their connections!

## What it does

The Concept Explorer takes concepts (like "Democracy", "Bitcoin", "Photosynthesis") and applies various transformations to them:
- **Scale Up**: Makes concepts global/systemic
- **Invert**: Creates the opposite
- **Lift**: Meta-level abstraction
- **Recurse**: Self-referential versions
- **Rotate**: Applies concept to different domains
- **Blend**: Mashes two concepts together

It then generates an interactive network graph showing how concepts evolve and connect!

## Setup

**Quick Start:**
```bash
./start.sh
```

This will automatically find an available port and start the server!

**Manual Start:**
```bash
source venv/bin/activate
python concept_explorer.py --port 5059
```

**Install as System Service:** See `INSTALL_SERVICE.md` for auto-start on boot.

## Usage

1. Enter a concept name (e.g., "Democracy", "Neural Network", "Evolution")
2. Choose a domain or leave it on "Auto-Detect"
3. Optionally add properties (comma-separated)
4. Click "Add Concept" or press Enter
5. Add multiple concepts if you want to blend them
6. Choose the number of expansion steps (1-3)
7. Click "GENERATE CONCEPT GALAXY"
8. Watch the interactive graph open in a new tab!

## Features

- Auto-detection of concept domains using the Datamuse API
- LLM integration for enhanced concept exploration
- Custom domain creation
- Interactive network visualization with physics simulation
- Color-coded nodes by domain
- Hover over nodes to see properties
- Click and drag to explore

## Requirements

See `requirements.txt` for all dependencies.

## Original Version

The original `GenEmp.py` was designed for Jupyter notebooks. This Flask version provides a standalone web interface for easier use!

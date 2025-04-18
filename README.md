# AI Research Assistant & Podcaster

A **FastAPI**-powered web application that generates comprehensive research reports and converts them into engaging podcast-style discussions using OpenAI's text-to-speech capabilities. Also allows you to download the report if reading is preferred.

---

## Overview

This application transforms research topics into dynamic content through multiple stages:

1. **Research Analysis**: Generates a team of AI analysts with diverse perspectives/personas
2. **Report Generation**: Creates a detailed research report with citations
3. **Podcast Conversion**: Transforms the report into a natural conversation
4. **Audio Synthesis**: Converts the podcast script into spoken audio using distinct voices for each analyst

### Research Graph Visualization

Below is a visualization of the research workflow and how different components interact:

![Research Graph](app/static/research_graph.png)

## Setup Instructions

**Clone and install**
```bash
git clone https://github.com/andrewlee977/scout.git
cd scout
poetry install
```

**Set up environment variables**
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key
NEWS_API_KEY=your_newsapi_key
TAVILY_API_KEY=your_tavily_key
```

Get your API keys from:
- [OpenAI](https://platform.openai.com/api-keys)
- [NewsAPI](https://newsapi.org/)
- [Tavily](https://tavily.com/)

**Run the app**
```bash
poetry run uvicorn app.main:app --reload
```


## Features

### Backend (FastAPI)
- Dynamic analyst generation with customizable roles and perspectives
- Research report generation with structured sections (introduction, insights, conclusion)
- Podcast script generation with natural dialogue
- OpenAI TTS integration with voice personality matching
- PDF report generation and download functionality

### Frontend
- Clean, responsive HTML interface
- Real-time loading indicators
- Custom audio player with progress tracking
- Interactive feedback system for analyst selection
- Download options for research reports

### Audio Processing
- Multi-voice podcast generation using OpenAI's TTS API
- Voice assignment based on analyst gender and role
- Custom voice instructions for personality matching
- Automatic audio segment combination with natural pauses

### Research Workflow
- Initial topic submission by the user
- Analyst team generation with feedback options
- Report generation with cited sources
- Podcast script creation
- Multi-voice audio synthesis

## Technical Features
- FastAPI for async request handling
- Jinja2 templating for dynamic HTML
- PDF report generation with FPDF
- Audio processing with pydub
- Structured logging system
- State management for research workflow

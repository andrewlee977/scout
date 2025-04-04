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

![Research Graph](static/research_graph.png)

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

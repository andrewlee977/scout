# Voice News Agent

A **FastAPI**-powered web application that fetches latest tech news articles with text input questions. Project takes this input to run an OpenAI LLM with a news fetcher tool and reads a summary using TTS.

---

## Overview

This application is designed to take text input and perform tasks like fetching news.

- **FastAPI** for the backend server  
- **HTML** for basic rendering the frontend pages  

By separating concerns—voice capture, speech-to-text conversion, and task delegation via different tools—this project aims to be **extensible**, **easy to integrate**, and **fun to use**.


---

## Features

- **FastAPI Backend**  
  Simple, async, and efficient server that processes requests and routes them to the correct functionalities.

- **HTML Frontend**  
  Provides a clean, server-rendered HTML interface for demoing or manual input.

- **Modular Tools**  
  - **NewsTool** for the latest headlines  

- **Agent-Oriented Design**  
  A central “agent” parses the text input and delegates tasks to the corresponding tools

FROM python:3.9

WORKDIR /code

# Install Poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock /code/

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev

# Install ffmpeg for audio processing
RUN apt-get update && apt-get install -y ffmpeg

# Install fonts for PDF generation
RUN apt-get install -y fonts-dejavu

# Copy the rest of the application
COPY . /code/

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

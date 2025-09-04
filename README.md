# Requirements:

- Install python version >= `python3.10`

- Install Task for pipeline (e.g, `brew install go-task` on Unix)

- Install Docker and Docker Compose for containerization

- Install Node.js version >= 20.19.0

- Install npm

- Ensure ports 5432, 8000, and 3000 are available (Docker) or 5173 (local development)

- For local development: `sudo` access for Docker commands (or add user to docker group)

## Supported Image Formats

- **Input formats**: JPEG, PNG, WebP, TIFF, BMP
- **Output**: PNG diff images with heatmap overlays
- **Processing**: OpenCV-based histogram comparison and pixel difference analysis
- **File size limit**: 50MB per image
- **Minimum file size**: 100 bytes

# Quick Start

**Recommended**: Use Docker for the easiest setup (no manual dependency installation required)

```bash
# 1. Check system requirements
task check-all

# 2. Start everything with Docker
task docker-up

# 3. Open the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000

# 4. Shutdown everything with Docker
task docker-down
```

### Check system requirements
`task check-all` - Run all system checks (Python, Node.js, Docker, ports)

### Other Docker Commands
- `task docker-build` - Build all Docker images (optional - docker-up builds automatically)
- `task docker-logs` - View logs from all services
- `task docker-clean` - Clean up Docker containers, images, and volumes
- `task docker-rebuild` - Rebuild and restart all services

### Docker Services
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **Database**: localhost:5432

### API Endpoints
- `POST /comparison` - Upload two images for comparison
- `GET /comparison/{id}` - Retrieve comparison results by ID
- `GET /health` - Health check endpoint
- `GET /static/diffs/{filename}` - Access generated diff images

# Local Development (Optional)

## Setup backend
`task backend-install`

## Setup Frontend
`task frontend-install`

## Start individual services
- **Postgres**: `task postgres-start`
- **Backend (FastAPI)**: `task server`
- **Frontend (React)**: `task frontend-dev`

## Start everything at once
- **Full stack (local)**: `task startup-full` - Frontend on port 5173

## For shutting down

- `task shutdown-full`

## For testing

`task test`

# Backend Design Choices

- PostgreSQL instance because I am familiar with it and it is the most popular production db tool
- Histogram pixel comparison - simple and intuitive approach appropriate for the time constraints
- Could move on to feature-based comparison such as a CNN if more time.
- Task pipelining + `.venv` environment for easy dependency and workflow management
- Unit tests for every function to ensure stability and generalisation capability
- Checked comparison of different image filetypes didn't affect the comparison score significantly
- A pixel difference threshold of 50 (99th percentile) and a minimum boxed pixel region size of 50 yielded good segmentation performance
- Generate images between sensitivity thresholds of 5 and 65 for front-end slider
- Blue channel comparison showed similar results to comparing the mean of all three channels
- Dockerised for portability 
- If I had more time I would do some load and memory testing

# Frontend Design Choices

- React + Vite for simplicity and faster build time -> avoid hand-writing CSS styling (more ergonomic)
- Tailwind for UI for easy component composition
- Save comparison history to localstorage in frontend (persists between pages refreshes) rather than using backend (need extra logic to prevent users from accessing other user's history, more complicated)
- Modular hierarchy -> root app component stores UploadForm component that does post/get to backend and feeds comparison data dowstream to HistoryList and ComparisonViewer, which stores images, diffs + library slider, diff scores.
- Drag and drop file upload boxes for QOL
- Dockerised for portability 
- More time -> prettier UI

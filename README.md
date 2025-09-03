# Requirements:

- Install python version >= `python3.10`

- Install Task for pipeline (e.g, `brew install go-task` on Unix)

- Install Docker.io for PostgreSQL database

- Install Node.js version >= 20.19.0

- Install npm

# To run:
## Check system requirements

`task check-versions`

## Setup backend

`task backend-install`

## Setup Frontend

`task frontend-install`

## Start individual services

- **Postgres**: `task postgres-start`
- **Backend (FastAPI)**: `task server`
- **Frontend (React)**: `task frontend-dev`

## Start everything at once

- **Backend only**: `task startup`
- **Full stack**: `task startup-full`

## For shutting down

`task shutdown-full`

## For testing

`task test`

# Backend Design Choices

- Docker-containerised PostgreSQL instance because I am familiar with it and it is the most popular production db tool
- Histogram pixel comparison - simple and intuitive approach appropriate for the time constraints
- Could move on to feature-based comparison such as a CNN if more time.
- Task pipelining + `.venv` environment for easy dependency and workflow management
- Unit tests for every function to ensure stability and generalisation capability
- Checked comparison of different image filetypes didn't affect the comparison score significantly
- A pixel difference threshold of 50 (99th percentile) and a minimum boxed pixel region size of 50 yielded good segmentation performance
- Generate images between sensitivity thresholds of 5 and 65 for front-end slider
- Blue channel comparison showed similar results to comparing the mean of all three channels
- If I had more time I would do some load and memory testing

# Frontend Design Choices

- React + Vite for simplicity and faster build time -> avoid hand-writing CSS styling (more ergonomic)
- Tailwind for UI for easy component composition
- Save comparison history to localstorage in frontend (persists between pages refreshes) rather than using backend (need extra logic to prevent users from accessing other user's history, more complicated)
- Modular hierarchy -> root app component stores UploadForm component that does post/get to backend and feeds comparison data dowstream to HistoryList and ComparisonViewer, which stores images, diffs + slider, diff scores.

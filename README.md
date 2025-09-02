# Requirements:

- Install python version >= `python3.10`

- Install Task for pipeline (e.g, `brew install go-task` on Unix)

- Install Docker.io for PostgreSQL database

# To run:
## Initialize project

`task install`

## Start Postgres

`task postgres:start`

## Run FastAPI

`task server`

## Or everything in one go

`task dev`

# Design choices

- Docker-containerised PostgreSQL instance because I am familiar with it and it is the most popular production db tool
- Histogram pixel comparison - simple and intuitive approach appropriate for the time constraints
- Could move on to feature-based comparison such as a CNN if more time.
- Task pipelining + `.venv` environment for easy dependency and workflow management
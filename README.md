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

`task startup`

## For shutting down

`task shutdown`

## and for testing

`task test`

# Design choices

- Docker-containerised PostgreSQL instance because I am familiar with it and it is the most popular production db tool
- Histogram pixel comparison - simple and intuitive approach appropriate for the time constraints
- Could move on to feature-based comparison such as a CNN if more time.
- Task pipelining + `.venv` environment for easy dependency and workflow management
- Unit tests for every function to ensure stability and generalisation capability
- Checked comparison of different image filetypes didn't affect the comparison score significantly
- A threshold of 50% difference and a minimum pixel region size of 50 yielded good segmentation performance
- Blue channel comparison showed similar results to comparing the mean of all three channels
- If I had more time I would do some load and memory testing
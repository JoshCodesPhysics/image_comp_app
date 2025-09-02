# Requirements:

- Install python version >= python3.10

- Install Task for pipeline (e.g, brew install go-task on Unix)

- Install Docker.io for PostgreSQL database

# To run:
## Initialize project

task install

## Start Postgres

task postgres:start

## Run FastAPI

task server

## Or everything in one go

task dev
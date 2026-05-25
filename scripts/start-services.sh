#!/bin/bash
set -e

ACTION="${1:-start}"

case "$ACTION" in
  start)
    echo "=== Starting services ==="
    docker compose up -d
    echo ""
    echo "Waiting for PostgreSQL to be ready..."
    docker compose exec -T postgres pg_isready -U postgres || \
      (echo "PostgreSQL not ready, waiting 10s..." && sleep 10 && docker compose exec -T postgres pg_isready -U postgres)
    echo ""
    echo "Services started:"
    echo "  Frontend (nginx):  http://localhost:8080"
    echo "  Backend (direct):  http://localhost:8000"
    echo "  Backend (Swagger): http://localhost:8000/docs"
    echo "  DB port:           localhost:5432"
    ;;
  stop)
    echo "=== Stopping services ==="
    docker compose down
    ;;
  restart)
    docker compose down
    sleep 2
    docker compose up -d
    ;;
  status)
    docker compose ps
    ;;
  logs)
    docker compose logs -f "${2:-}"
    ;;
  clean)
    echo "=== Cleaning up ==="
    docker compose down -v --remove-orphans
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs [service]|clean}"
    exit 1
    ;;
esac

# Parking Lot YOLO - Real-time Object Detection

A real-time object detection system using YOLOv8 with Django, Channels, and React.

## Prerequisites

### Install `uv` (Python Package Manager)

`uv` is a fast Python package installer. Install it from:
- [GitHub - astral-sh/uv](https://github.com/astral-sh/uv)
- Or use: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Install Redis

Redis is required for Django Channels to handle WebSocket connections.

#### macOS
```bash
# Install via Homebrew
brew install redis

# Start Redis on default port 6379
redis-server

# Or start on a custom port (e.g., 6380)
redis-server --port 6380
```

#### Linux (Ubuntu/Debian)
```bash
# Install
sudo apt-get install redis-server

# Start
redis-server

# Or start on a custom port
redis-server --port 6380
```

#### Docker
```bash
# Run Redis in a container on port 6379
docker run -d -p 6379:6379 redis:latest

# Or on a custom port
docker run -d -p 6380:6379 redis:latest
```

See [Redis Documentation](https://redis.io/documentation) for more details.

## Backend Setup

### 1. Configure Virtual Environment

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 2. Install Backend Dependencies

```bash
# Using uv
uv pip install -r pyproject.toml

# Or using pip
pip install -e .
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Django Settings
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Redis Configuration (default: localhost on port 6379)
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# WebSocket Configuration (for development)
WS_SCHEME=ws
WS_HOST=localhost:8000
```

You can copy `.env.example` if provided, or use the `.env.local` format for local overrides.

### 4. Run Django Server

```bash
# Apply migrations (first time)
python manage.py migrate

# Start the Django development server
python manage.py runserver
# Or specify a custom port
python manage.py runserver 8000
```

The backend will be available at `http://localhost:8000`.

## Frontend Setup

### 1. Install Node Dependencies

Navigate to the frontend directory and install dependencies:

```bash
cd frontend

# Install dependencies
npm install
```

### 2. Configure Environment Variables

Create a `.env.local` file in the `frontend/` directory:

```bash
# WebSocket Configuration
VITE_WS_SCHEME=ws
VITE_WS_HOST=localhost:8000
```

For production, use:
```bash
VITE_WS_SCHEME=wss
VITE_WS_HOST=yourdomain.com
```

Copy `.env.example` for a template.

### 3. Run Frontend Development Server

```bash
# From the frontend directory
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Production Deployment

For deploying over a network or using tunnels, see:
- [Cloudflare Tunnel Setup Guide](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/#2b-connect-a-network)

This allows you to expose your local development server securely over HTTPS without opening ports.

## Environment Variables Reference

### Backend (.env)
- `SECRET_KEY`: Django secret key (generate a new one for production)
- `DEBUG`: Set to `False` in production
- `ALLOWED_HOSTS`: Comma-separated list of allowed hosts
- `REDIS_HOST`: Redis server host (default: `127.0.0.1`)
- `REDIS_PORT`: Redis server port (default: `6379`)
- `WS_SCHEME`: WebSocket scheme (`ws` for development, `wss` for production)
- `WS_HOST`: WebSocket host (e.g., `localhost:8000` or `yoloback.example.com`)

### Frontend (.env.local)
- `VITE_WS_SCHEME`: WebSocket scheme (`ws` for development, `wss` for production)
- `VITE_WS_HOST`: WebSocket host (e.g., `localhost:8000` or `yoloback.example.com`)

## Troubleshooting

### Redis Connection Error
- Ensure Redis is running: `redis-cli ping` should return `PONG`
- Check `REDIS_HOST` and `REDIS_PORT` in `.env`

### WebSocket Connection Error
- Verify backend is running
- Check `VITE_WS_SCHEME` and `VITE_WS_HOST` in frontend `.env.local`
- Ensure both backend and frontend are properly configured for the same host

### Port Already in Use
- Change the port when running Django: `python manage.py runserver 8001`
- Change Redis port: `redis-server --port 6380` and update `.env`
- Update `.env` files accordingly

## Technologies

- **Backend**: Django, Django Channels, Redis, YOLOv8
- **Frontend**: React, Vite, Chakra UI
- **Real-time Communication**: WebSockets
- **Object Detection**: YOLO (Ultralytics)

#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   🚀 Starting Sailor Development Stack   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down all services...${NC}"
    kill $DENSE_PID $SPARSE_PID $BACKEND_PID 2>/dev/null
    wait $DENSE_PID $SPARSE_PID $BACKEND_PID 2>/dev/null
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT TERM

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠️  'uv' is not installed. Install it with: pip install uv${NC}"
    exit 1
fi

echo -e "${GREEN}📦 Installing dependencies...${NC}"
echo ""

# Install dense embedding service dependencies
echo -e "${BLUE}→${NC} Dense Embedding Service dependencies..."
cd services/dense-embedding
if [ ! -d ".venv" ]; then
    uv venv --quiet
fi
uv pip install -r requirements.txt --quiet
cd ../..

# Install sparse embedding service dependencies
echo -e "${BLUE}→${NC} Sparse Embedding Service dependencies..."
cd services/sparse-embedding
if [ ! -d ".venv" ]; then
    uv venv --quiet
fi
uv pip install -r requirements.txt --quiet
cd ../..

# Install backend dependencies
echo -e "${BLUE}→${NC} Backend dependencies..."
cd backend
if [ ! -d ".venv" ]; then
    uv venv --quiet
fi
uv pip install -e . --quiet
cd ..

echo ""
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Start Dense Embedding Service
echo -e "${BLUE}🧠 Starting Dense Embedding Service (port 8001)...${NC}"
cd services/dense-embedding
uv run python app.py > ../../logs/dense-embedding.log 2>&1 &
DENSE_PID=$!
cd ../..

# Wait a bit for it to start
sleep 2

# Start Sparse Embedding Service
echo -e "${BLUE}🔍 Starting Sparse Embedding Service (port 8002)...${NC}"
cd services/sparse-embedding
uv run python app.py > ../../logs/sparse-embedding.log 2>&1 &
SPARSE_PID=$!
cd ../..

# Wait for services to initialize
sleep 3

# Start Backend API
echo -e "${BLUE}🌐 Starting FastAPI Backend (port 8000)...${NC}"
cd backend
uv run uvicorn app.main:app --reload --port 8000 > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 2

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     ✅ All Services Started Successfully  ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📍 Service URLs:${NC}"
echo -e "   • Backend API:          ${GREEN}http://localhost:8000${NC}"
echo -e "   • Dense Embedding:      ${GREEN}http://localhost:8001${NC}"
echo -e "   • Sparse Embedding:     ${GREEN}http://localhost:8002${NC}"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo -e "   • Backend Docs:         ${GREEN}http://localhost:8000/docs${NC}"
echo -e "   • Dense Embedding Docs: ${GREEN}http://localhost:8001/docs${NC}"
echo -e "   • Sparse Embedding Docs:${GREEN}http://localhost:8002/docs${NC}"
echo ""
echo -e "${BLUE}📋 Logs:${NC}"
echo -e "   • Backend:              ${YELLOW}logs/backend.log${NC}"
echo -e "   • Dense Embedding:      ${YELLOW}logs/dense-embedding.log${NC}"
echo -e "   • Sparse Embedding:     ${YELLOW}logs/sparse-embedding.log${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for user interrupt
wait

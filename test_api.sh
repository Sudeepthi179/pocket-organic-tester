#!/bin/bash
# Test script for Pocket Organic Tester API

echo "============================================================"
echo "Testing Pocket Organic Tester API"
echo "============================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Start Flask server in background
echo "Starting Flask server..."
source venv/bin/activate
python app.py > /tmp/flask_test.log 2>&1 &
FLASK_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 3

# Test 1: Root endpoint
echo "Test 1: Root endpoint (GET /)"
RESPONSE=$(curl -s http://localhost:5000/)
if echo "$RESPONSE" | grep -q "Pocket Organic Tester API is running"; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi
echo ""

# Test 2: Health check
echo "Test 2: Health check (GET /api/health)"
RESPONSE=$(curl -s http://localhost:5000/api/health)
if echo "$RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi
echo ""

# Test 3: Predict Apple (Organic)
echo "Test 3: Predict Apple (Organic)"
RESPONSE=$(curl -s -X POST http://localhost:5000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"spectral_values": [0.47, 0.55, 0.60, 0.64, 0.57, 0.50, 0.45, 0.41]}')
if echo "$RESPONSE" | grep -q "Apple"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "   Response: $RESPONSE"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "   Response: $RESPONSE"
fi
echo ""

# Test 4: Predict Banana (Non-Organic)
echo "Test 4: Predict Banana (Non-Organic)"
RESPONSE=$(curl -s -X POST http://localhost:5000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"spectral_values": [0.72, 0.78, 0.82, 0.85, 0.80, 0.75, 0.68, 0.62]}')
if echo "$RESPONSE" | grep -q "Banana"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "   Response: $RESPONSE"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "   Response: $RESPONSE"
fi
echo ""

# Test 5: Predict Tomato (Organic)
echo "Test 5: Predict Tomato (Organic)"
RESPONSE=$(curl -s -X POST http://localhost:5000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"spectral_values": [0.71, 0.44, 0.37, 0.40, 0.48, 0.55, 0.51, 0.46]}')
if echo "$RESPONSE" | grep -q "Tomato"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "   Response: $RESPONSE"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "   Response: $RESPONSE"
fi
echo ""

# Test 6: Error handling - wrong number of values
echo "Test 6: Error handling - wrong number of values"
RESPONSE=$(curl -s -X POST http://localhost:5000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"spectral_values": [0.45, 0.52, 0.58, 0.62, 0.55, 0.48, 0.42]}')
if echo "$RESPONSE" | grep -q "exactly 8 values"; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi
echo ""

# Stop Flask server
echo "Stopping Flask server..."
kill $FLASK_PID 2>/dev/null

echo ""
echo "============================================================"
echo "Testing Complete!"
echo "============================================================"

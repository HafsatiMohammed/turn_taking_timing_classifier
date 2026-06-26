#!/bin/bash
# Run complete experiment: train and evaluate model
# Usage: ./scripts/run_experiment.sh [ablation_name]

set -e

ABLATION="${1:-ablation_1}"
CONFIG_FILE="configs/${ABLATION}.yaml"
CHECKPOINT_DIR="checkpoints/${ABLATION}"
RESULTS_FILE="results_${ABLATION}.json"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Turn-Taking Timing Classifier Experiment${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Ablation: $ABLATION"
echo "Config: $CONFIG_FILE"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

echo -e "${BLUE}[1/3] Starting Training...${NC}"
python scripts/train.py --config "$CONFIG_FILE"

# Find best checkpoint
BEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_best.pt 2>/dev/null | head -1)

if [ -z "$BEST_CHECKPOINT" ]; then
    echo "Error: No best checkpoint found"
    exit 1
fi

echo -e "${GREEN}✓ Training complete${NC}"
echo "Best checkpoint: $BEST_CHECKPOINT"
echo ""

echo -e "${BLUE}[2/3] Running Evaluation...${NC}"
python scripts/test.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$BEST_CHECKPOINT" \
    --output "$RESULTS_FILE"

echo -e "${GREEN}✓ Evaluation complete${NC}"
echo "Results: $RESULTS_FILE"
echo ""

echo -e "${BLUE}[3/3] Experiment Summary${NC}"
python -c "
import json
with open('$RESULTS_FILE') as f:
    results = json.load(f)['metrics']
print(f\"Macro-F1: {results.get('macro_f1', 0):.4f}\")
print(f\"BACKCHANNEL F1: {results.get('backchannel_f1', 0):.4f}\")
print(f\"START_SPEAKING F1: {results.get('start_speaking_f1', 0):.4f}\")
print(f\"False Entry Rate: {results.get('false_entry_rate', 0):.4f}\")
print(f\"Missed Entry Rate: {results.get('missed_entry_rate', 0):.4f}\")
print(f\"ECE: {results.get('ece', 0):.4f}\")
print(f\"Inference Latency: {results.get('inference_latency_ms', 0):.2f}ms\")
"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Experiment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

#!/bin/bash

# Default values
CONFIG="config/config.json"
TASK="test"
NUM_EPISODES=2

# Help function
function show_help {
    echo "Usage: $0 [OPTIONS]"
    echo "Record robot trajectory data"
    echo ""
    echo "Options:"
    echo "  -c, --config        Configuration file path (default: config/config.json)"
    echo "  -t, --task          Task name (default: test)"
    echo "  -n, --num-episodes  Number of episodes to record (default: 2)"
    echo "  -h, --help          Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -n|--num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Check if config file and project interpreter exist
if [ ! -f "$CONFIG" ]; then
    echo "Error: Configuration file '$CONFIG' not found!"
    exit 1
fi
if [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Error: uv environment missing. Run 'uv venv --python 3.10 .venv' first!" >&2
    exit 1
fi

# Execute the Python script
exec "$PROJECT_ROOT/.venv/bin/python" data_collection.py \
    --config "$CONFIG" \
    --task "$TASK" \
    --num_episodes "$NUM_EPISODES"

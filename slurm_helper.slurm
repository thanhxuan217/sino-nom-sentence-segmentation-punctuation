#!/bin/bash

# ============================================================================
# Helper Script for SLURM Job Management
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: ./slurm_helper.slurm [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  submit [script]     Submit a SLURM job"
    echo "  status              Check status of your jobs"
    echo "  cancel [job_id]     Cancel a job"
    echo "  logs [job_id]       View logs for a job"
    echo "  tail [job_id]       Tail logs in real-time"
    echo "  info [job_id]       Get detailed job information"
    echo "  queue               Show queue information"
    echo "  cleanup             Clean up old log files"
    echo ""
    echo "Examples:"
    echo "  ./slurm_helper.slurm submit run.slurm"
    echo "  ./slurm_helper.slurm status"
    echo "  ./slurm_helper.slurm tail 12345"
    echo "  ./slurm_helper.slurm cancel 12345"
}

# Function to submit a job
submit_job() {
    local script=$1
    if [ -z "$script" ]; then
        echo -e "${RED}Error: No script specified${NC}"
        echo "Usage: ./slurm_helper.slurm submit <script.slurm>"
        exit 1
    fi
    
    if [ ! -f "$script" ]; then
        echo -e "${RED}Error: Script '$script' not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Submitting job: $script${NC}"
    sbatch $script
}

# Function to check job status
check_status() {
    echo -e "${GREEN}Your running/pending jobs:${NC}"
    squeue -u $USER --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
}

# Function to cancel a job
cancel_job() {
    local job_id=$1
    if [ -z "$job_id" ]; then
        echo -e "${RED}Error: No job ID specified${NC}"
        echo "Usage: ./slurm_helper.slurm cancel <job_id>"
        exit 1
    fi
    
    echo -e "${YELLOW}Cancelling job: $job_id${NC}"
    scancel $job_id
    echo -e "${GREEN}Job $job_id cancelled${NC}"
}

# Function to view logs
view_logs() {
    local job_id=$1
    if [ -z "$job_id" ]; then
        echo -e "${RED}Error: No job ID specified${NC}"
        echo "Usage: ./slurm_helper.slurm logs <job_id>"
        exit 1
    fi
    
    local out_file="logs/slurm_${job_id}.out"
    local err_file="logs/slurm_${job_id}.err"
    
    if [ -f "$out_file" ]; then
        echo -e "${GREEN}=== Output log: $out_file ===${NC}"
        cat $out_file
    else
        echo -e "${RED}Output log not found: $out_file${NC}"
    fi
    
    echo ""
    
    if [ -f "$err_file" ]; then
        echo -e "${YELLOW}=== Error log: $err_file ===${NC}"
        cat $err_file
    else
        echo -e "${GREEN}No errors logged${NC}"
    fi
}

# Function to tail logs in real-time
tail_logs() {
    local job_id=$1
    if [ -z "$job_id" ]; then
        echo -e "${RED}Error: No job ID specified${NC}"
        echo "Usage: ./slurm_helper.slurm tail <job_id>"
        exit 1
    fi
    
    local out_file="logs/slurm_${job_id}.out"
    
    if [ -f "$out_file" ]; then
        echo -e "${GREEN}Tailing output log: $out_file${NC}"
        echo "Press Ctrl+C to stop"
        tail -f $out_file
    else
        echo -e "${RED}Output log not found: $out_file${NC}"
        echo "Waiting for log file to be created..."
        
        # Wait for log file to be created (up to 60 seconds)
        for i in {1..60}; do
            if [ -f "$out_file" ]; then
                echo -e "${GREEN}Log file created! Starting tail...${NC}"
                tail -f $out_file
                break
            fi
            sleep 1
        done
        
        if [ ! -f "$out_file" ]; then
            echo -e "${RED}Log file not created after 60 seconds${NC}"
            exit 1
        fi
    fi
}

# Function to get detailed job info
job_info() {
    local job_id=$1
    if [ -z "$job_id" ]; then
        echo -e "${RED}Error: No job ID specified${NC}"
        echo "Usage: ./slurm_helper.slurm info <job_id>"
        exit 1
    fi
    
    echo -e "${GREEN}Job Information for $job_id:${NC}"
    scontrol show job $job_id
}

# Function to show queue information
show_queue() {
    echo -e "${GREEN}Cluster Queue Status:${NC}"
    squeue --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" | head -n 20
    
    echo ""
    echo -e "${GREEN}Your Jobs:${NC}"
    squeue -u $USER --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
}

# Function to cleanup old logs
cleanup_logs() {
    echo -e "${YELLOW}Cleaning up log files older than 7 days...${NC}"
    
    if [ -d "logs" ]; then
        old_files=$(find logs -name "slurm_*.out" -mtime +7 2>/dev/null)
        old_err_files=$(find logs -name "slurm_*.err" -mtime +7 2>/dev/null)
        
        count=0
        for file in $old_files $old_err_files; do
            if [ -f "$file" ]; then
                rm "$file"
                ((count++))
            fi
        done
        
        echo -e "${GREEN}Removed $count old log files${NC}"
    else
        echo -e "${YELLOW}No logs directory found${NC}"
    fi
}

# Main script logic
case "$1" in
    submit)
        submit_job "$2"
        ;;
    status)
        check_status
        ;;
    cancel)
        cancel_job "$2"
        ;;
    logs)
        view_logs "$2"
        ;;
    tail)
        tail_logs "$2"
        ;;
    info)
        job_info "$2"
        ;;
    queue)
        show_queue
        ;;
    cleanup)
        cleanup_logs
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        usage
        exit 1
        ;;
esac

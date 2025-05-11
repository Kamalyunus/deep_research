#!/usr/bin/env python3
"""
Script to automatically resolve git merge conflicts by taking the upstream changes.
"""
import os
import re
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pattern to match git merge conflict markers
CONFLICT_PATTERN = re.compile(r'<<<<<<< .*?\n(.*?)=======\n(.*?)>>>>>>> .*?\n', re.DOTALL)

def resolve_conflicts(file_path):
    """
    Resolve conflicts in a single file by keeping the first part (upstream).
    
    Args:
        file_path: Path to the file with conflicts
    
    Returns:
        bool: True if conflicts were found and resolved, False otherwise
    """
    logger.info(f"Processing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '<<<<<<< ' not in content:
        logger.info(f"No conflicts found in {file_path}")
        return False
    
    def replace_conflict(match):
        # Take the first part (Updated upstream)
        return match.group(1)
    
    new_content = CONFLICT_PATTERN.sub(replace_conflict, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    logger.info(f"Conflicts resolved in {file_path}")
    return True

def main():
    """
    Find and resolve all conflicts in Python files.
    """
    # Find all Python files with conflicts
    python_files = glob.glob('./deep_research/**/*.py', recursive=True)
    conflicts_resolved = 0
    
    for file_path in python_files:
        try:
            if resolve_conflicts(file_path):
                conflicts_resolved += 1
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Resolved conflicts in {conflicts_resolved} files")

if __name__ == "__main__":
    main()